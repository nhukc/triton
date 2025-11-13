from typing import Sequence, List, TypeVar, Tuple, Callable
import math
import os
from triton.language.semantic import TritonSemantic
from . import _core as ttgl
from ._layouts import AutoLayout, DistributedLayout, DistributedLinearLayout, SliceLayout, SharedLayout
from triton._C.libtriton.gluon_ir import GluonOpBuilder, compute_tmem_reg_layout
from triton.compiler.code_generator import flatten_values_to_ir, unflatten_ir_values

# Debug flag to insert barriers after every operation
_GLUON_INSERT_BARRIERS = os.environ.get("GLUON_INSERT_BARRIERS", "0") == "1"

TensorTy = TypeVar("TensorTy")


def _check(cond: bool, msg_fn: Callable[[], str], category=ValueError):
    if not cond:
        raise category(msg_fn())


def _is_int_list(value):
    return isinstance(value, Sequence) and all(isinstance(i, int) for i in value)


def _compute_tmem_reg_layout(element_ty, shape, layout, num_warps, instr_variant, ctas_per_cga, cta_split_num,
                             cta_order):
    _check(isinstance(instr_variant, str), lambda: "instr_variant must be a string")
    _check(instr_variant in ("32x32b", "16x64b", "16x128b", "16x256b", "16x32bx2", "32x32b_splitn"),
           lambda: f"unknown instr_variant: {instr_variant}")
    _check(isinstance(num_warps, int), lambda: f"num_warps must be an int but got {type(num_warps)!r}")
    _check(num_warps >= 4 and (num_warps & (num_warps - 1)) == 0, lambda: "num_warps must be a power of two and >= 4")

    shape = list(shape)
    _check(all(isinstance(dim, int) for dim in shape), lambda: f"shape entries must be ints but got {shape}")
    rank = len(shape)
    _check(rank == 2, lambda: "expected a 2D tensor")

    ctas_per_cga = list(ctas_per_cga)
    cta_split_num = list(cta_split_num)
    cta_order = list(cta_order)
    splitn = instr_variant == "32x32b_splitn"
    atom_variant = "32x32b" if splitn else instr_variant

    _check(len(ctas_per_cga) == rank, lambda: "ctas_per_cga rank mismatch")
    _check(len(cta_split_num) == rank, lambda: "cta_split_num rank mismatch")
    _check(len(cta_order) == rank, lambda: "cta_order rank mismatch")

    layout_obj = compute_tmem_reg_layout(
        element_ty,
        shape,
        layout,
        num_warps,
        atom_variant,
        ctas_per_cga,
        cta_split_num,
        cta_order,
    )
    _check(layout_obj is not None,
           lambda: f"TMEM layout '{atom_variant}' unsupported for shape {shape} and num_warps {num_warps}")

    if splitn:
        N = shape[1]
        if not layout_obj.reg_bases:
            # We cannot use this layout in a load or a store ATM due to a PTX bug!
            # You can work around this by loading to 32x32b and follow by a convert_layout to this layout.
            _check(layout_obj.lane_bases[-1] == [0, N // 2],
                   lambda: f"splitn with 1 register requires the last lane basis to be [0, N / 2]. Got {layout_obj}")
            layout_obj.reg_bases.append([0, N // 2])
            layout_obj.lane_bases[-1] = [0, 0]
        elif layout_obj.reg_bases[-1] != [0, N // 2]:
            bitwidth = element_ty.primitive_bitwidth
            _check(
                len(layout_obj.reg_bases) * bitwidth > 32,
                lambda: "splitn requires register bases of more than 2 32 bit registers")

            reg_bases = layout_obj.reg_bases
            for bases_str in ("lane_bases", "warp_bases"):
                bases = getattr(layout_obj, bases_str)
                for i, basis in enumerate(bases):
                    if basis == [0, N // 2]:
                        reg_bases[-1], bases[i] = bases[i], reg_bases[-1]
                        return layout_obj
            assert False, f"splitn requires at least one basis of [0, N / 2]. Got {layout}"
    return layout_obj


_compute_tmem_reg_layout.__triton_builtin__ = True


class GluonCallerContext:

    def __init__(self, num_warps: int):
        self.num_warps = num_warps

    def mangle(self):
        return f"_NW{self.num_warps}"

    def initialize_callee(self, fn, builder):
        fn.set_attr("ttg.num-warps", builder.get_int32_attr(self.num_warps))


class Channel:
    """Multi-buffered channel for producer-consumer communication.

    This class only stores the shared buffers and barriers.
    Each sender/receiver maintains its own counter.

    Supports multiple producers and/or multiple consumers:
    - Multiple producers: each producer gets buffers at (counter * num_producers + producer_id) % num_buffers
    - Multiple consumers: each consumer gets buffers at (counter * num_consumers + consumer_id) % num_buffers
    """

    def __init__(self, semantic, num_buffers, shapes, dtypes, layouts, num_producers, num_consumers):
        self.semantic = semantic
        self.num_buffers = num_buffers
        self.shapes = shapes
        self.dtypes = dtypes
        self.layouts = layouts
        self.num_producers = num_producers
        self.num_consumers = num_consumers

        # Allocate buffers for each tensor in the bundle
        self.buffers = []
        for shape, dtype, layout in zip(shapes, dtypes, layouts):
            full_shape = [num_buffers] + list(shape)
            buf = semantic.allocate_shared(dtype, full_shape, layout, None)

            # Initialize all buffer slots with sentinel value (-999999)
            # This helps detect uninitialized reads during debugging
            if dtype == ttgl.int32:
                sentinel = -999999
            elif dtype == ttgl.float16 or dtype == ttgl.float32:
                sentinel = -999999.0
            else:
                sentinel = -999999

            # Create a simple blocked layout for initialization
            # Use all available warps and threads to initialize in parallel
            num_warps = semantic.builder.options.num_warps
            if len(shape) == 1:
                init_layout = ttgl.BlockedLayout([1], [32], [num_warps], [0])
            else:
                # For 2D tensors, use a layout that distributes work across all threads
                init_layout = ttgl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])

            for i in range(num_buffers):
                buf_idx = semantic.memdesc_index(buf, ttgl.constexpr(i))
                sentinel_val = semantic.full(shape, sentinel, dtype, init_layout)
                semantic.shared_store(buf_idx, sentinel_val)

            self.buffers.append(buf)

        # Allocate barriers using hopper mbarrier
        from triton.experimental.gluon.language.nvidia.hopper.mbarrier import MBarrierLayout
        bar_layout = MBarrierLayout()
        self.empty_barriers = semantic.allocate_shared(ttgl.int64, [num_buffers, 1], bar_layout, None)
        self.ready_barriers = semantic.allocate_shared(ttgl.int64, [num_buffers, 1], bar_layout, None)

        # Initialize barriers
        for i in range(num_buffers):
            empty_idx = semantic.memdesc_index(self.empty_barriers, ttgl.constexpr(i))
            ready_idx = semantic.memdesc_index(self.ready_barriers, ttgl.constexpr(i))
            semantic.builder.create_mbarrier_init(empty_idx.handle, 1)
            semantic.builder.create_mbarrier_init(ready_idx.handle, 1)

    def sender(self, producer_id=0):
        return ChannelSender(self, producer_id)

    def receiver(self, consumer_id=0):
        return ChannelReceiver(self, consumer_id)


class ChannelSenderType(ttgl.base_type):
    """Type for ChannelSender."""

    def __init__(self, num_buffers, shapes, dtypes, layouts, buffer_types, barrier_types, num_producers, producer_id):
        self.num_buffers = num_buffers
        self.shapes = shapes
        self.dtypes = dtypes
        self.layouts = layouts
        self.buffer_types = buffer_types
        self.barrier_types = barrier_types
        self.num_producers = num_producers
        self.producer_id = producer_id

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.num_buffers == other.num_buffers and
                self.buffer_types == other.buffer_types and
                self.barrier_types == other.barrier_types and
                self.num_producers == other.num_producers and
                self.producer_id == other.producer_id)

    def mangle(self):
        """Generate a unique type signature for this channel sender."""
        buf_mangles = "_".join(t.mangle() for t in self.buffer_types)
        bar_mangles = "_".join(t.mangle() for t in self.barrier_types)
        return f"ChSnd_{self.num_buffers}_P{self.num_producers}_{self.producer_id}_{buf_mangles}_{bar_mangles}"

    def _flatten_ir_types(self, builder, out):
        """Flatten to IR types by passing all buffer and barrier types."""
        for buf_type in self.buffer_types:
            buf_type._flatten_ir_types(builder, out)
        for bar_type in self.barrier_types:
            bar_type._flatten_ir_types(builder, out)
        # Add counter type (int32 scalar)
        out.append(builder.get_int32_ty())

    def _unflatten_ir(self, handles, cursor):
        """Reconstruct ChannelSender from IR handles."""
        # First unflatten all buffers
        buffers = []
        for buf_type in self.buffer_types:
            buf, cursor = buf_type._unflatten_ir(handles, cursor)
            buffers.append(buf)

        # Then unflatten barriers
        empty_barriers, cursor = self.barrier_types[0]._unflatten_ir(handles, cursor)
        ready_barriers, cursor = self.barrier_types[1]._unflatten_ir(handles, cursor)

        # Unflatten counter (it's an int32 scalar)
        from triton.experimental.gluon.language import int32
        counter_type = ttgl.tensor([1], int32).type
        counter, cursor = counter_type._unflatten_ir(handles, cursor)

        # Reconstruct the sender
        sender = ChannelSender.__new__(ChannelSender)
        sender.num_buffers = self.num_buffers
        sender.shapes = self.shapes
        sender.dtypes = self.dtypes
        sender.layouts = self.layouts
        sender.buffers = buffers
        sender.empty_barriers = empty_barriers
        sender.ready_barriers = ready_barriers
        sender.counter = counter
        sender.num_producers = self.num_producers
        sender.producer_id = self.producer_id
        sender.type = self

        return sender, cursor


class ChannelSender(ttgl.base_value):
    """Producer side of a channel.

    Maintains its own counter that is only modified by the sender.
    """

    def __init__(self, channel: Channel, producer_id: int):
        self.num_buffers = channel.num_buffers
        self.shapes = channel.shapes
        self.dtypes = channel.dtypes
        self.layouts = channel.layouts
        self.buffers = channel.buffers
        self.empty_barriers = channel.empty_barriers
        self.ready_barriers = channel.ready_barriers
        self.num_producers = channel.num_producers
        self.producer_id = producer_id

        # Create the type
        buffer_types = [buf.type for buf in self.buffers]
        barrier_types = [channel.empty_barriers.type, channel.ready_barriers.type]
        self.type = ChannelSenderType(
            channel.num_buffers, channel.shapes, channel.dtypes, channel.layouts,
            buffer_types, barrier_types, channel.num_producers, producer_id
        )

        # Counter is an IR tensor that will be updated across calls
        self.counter = channel.semantic.to_tensor(ttgl.constexpr(0))

    def _flatten_ir(self, handles):
        """Flatten to IR by passing all buffers and barriers."""
        # Flatten all buffers
        for buf in self.buffers:
            buf._flatten_ir(handles)
        # Flatten barriers
        self.empty_barriers._flatten_ir(handles)
        self.ready_barriers._flatten_ir(handles)
        # Flatten counter
        self.counter._flatten_ir(handles)

    @ttgl.builtin
    def allocate(self, _semantic=None):
        """Allocate buffer(s) for writing. Waits if all buffers are full.
        Returns (buffers_tuple, updated_sender) to make state change explicit.
        """
        semantic = _semantic
        num_buffers = self.num_buffers
        counter = self.counter

        # Compute idx = (counter * num_producers + producer_id) % num_buffers
        num_bufs_tensor = semantic.to_tensor(ttgl.constexpr(num_buffers))
        num_prods_tensor = semantic.to_tensor(ttgl.constexpr(self.num_producers))
        prod_id_tensor = semantic.to_tensor(ttgl.constexpr(self.producer_id))

        temp = semantic.mul(counter, num_prods_tensor, sanitize_overflow=False)
        temp = semantic.add(temp, prod_id_tensor, sanitize_overflow=False)
        idx_tensor = semantic.mod(temp, num_bufs_tensor)

        # Compute phase = (counter // (num_buffers // num_producers)) & 1, then XOR with 1 (wait for empty)
        # Each producer revisits the same buffer slot every (num_buffers // num_producers) increments
        one_tensor = semantic.to_tensor(ttgl.constexpr(1))
        buffers_per_producer = num_buffers // self.num_producers
        bufs_per_prod_tensor = semantic.to_tensor(ttgl.constexpr(buffers_per_producer))
        div_result = semantic.floordiv(counter, bufs_per_prod_tensor)
        and_result = semantic.and_(div_result, one_tensor)
        phase_tensor = semantic.xor_(and_result, one_tensor)

        # Wait for empty barrier
        empty_idx = semantic.memdesc_index(self.empty_barriers, idx_tensor)
        pred_tensor = semantic.to_tensor(ttgl.constexpr(True))
        semantic.builder.create_mbarrier_wait(empty_idx.handle, phase_tensor.handle, pred_tensor.handle, [])

        # Index into each buffer and return
        result = []
        for buf in self.buffers:
            buf_idx = semantic.memdesc_index(buf, idx_tensor)
            result.append(buf_idx)

        # Increment counter (only sender modifies this)
        new_counter = semantic.add(counter, one_tensor, sanitize_overflow=False)

        # Append channel metadata as the last two elements of the tuple
        # This ensures they survive compiler transformations
        result.append(idx_tensor)
        result.append(counter)

        # Create updated sender with new counter
        updated_sender = ChannelSender.__new__(ChannelSender)
        updated_sender.num_buffers = self.num_buffers
        updated_sender.shapes = self.shapes
        updated_sender.dtypes = self.dtypes
        updated_sender.layouts = self.layouts
        updated_sender.buffers = self.buffers
        updated_sender.empty_barriers = self.empty_barriers
        updated_sender.ready_barriers = self.ready_barriers
        updated_sender.num_producers = self.num_producers
        updated_sender.producer_id = self.producer_id
        updated_sender.counter = new_counter
        updated_sender.type = self.type

        return ttgl.tuple([ttgl.tuple(result), updated_sender])

    @ttgl.builtin
    def send(self, buffers, _semantic=None):
        """Signal that buffer(s) are ready for consumption.

        Args:
            buffers: The buffers returned from allocate() - a tuple where the last two elements
                    are the channel index and counter
        """
        semantic = _semantic

        # Extract the channel metadata from the last two elements of the tuple
        # The tuple structure from allocate() is: [buffer0, buffer1, ..., idx_tensor, counter]
        idx_tensor = buffers[-2]
        counter = buffers[-1]

        # Compute phase = (counter // (num_buffers // num_producers)) & 1 (for ready barrier)
        num_bufs_tensor = semantic.to_tensor(ttgl.constexpr(self.num_buffers))
        one_tensor = semantic.to_tensor(ttgl.constexpr(1))
        buffers_per_producer = self.num_buffers // self.num_producers
        bufs_per_prod_tensor = semantic.to_tensor(ttgl.constexpr(buffers_per_producer))
        div_result = semantic.floordiv(counter, bufs_per_prod_tensor)
        phase_tensor = semantic.and_(div_result, one_tensor)

        # Signal ready barrier
        ready_idx = semantic.memdesc_index(self.ready_barriers, idx_tensor)
        pred_tensor = semantic.to_tensor(ttgl.constexpr(True))
        semantic.builder.create_fence_async_shared(False)
        semantic.builder.create_mbarrier_arrive(ready_idx.handle, 1, pred_tensor.handle)


class ChannelReceiverType(ttgl.base_type):
    """Type for ChannelReceiver."""

    def __init__(self, num_buffers, shapes, dtypes, layouts, buffer_types, barrier_types, num_consumers, consumer_id):
        self.num_buffers = num_buffers
        self.shapes = shapes
        self.dtypes = dtypes
        self.layouts = layouts
        self.buffer_types = buffer_types
        self.barrier_types = barrier_types
        self.num_consumers = num_consumers
        self.consumer_id = consumer_id

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.num_buffers == other.num_buffers and
                self.buffer_types == other.buffer_types and
                self.barrier_types == other.barrier_types and
                self.num_consumers == other.num_consumers and
                self.consumer_id == other.consumer_id)

    def mangle(self):
        """Generate a unique type signature for this channel receiver."""
        buf_mangles = "_".join(t.mangle() for t in self.buffer_types)
        bar_mangles = "_".join(t.mangle() for t in self.barrier_types)
        return f"ChRcv_{self.num_buffers}_C{self.num_consumers}_{self.consumer_id}_{buf_mangles}_{bar_mangles}"

    def _flatten_ir_types(self, builder, out):
        """Flatten to IR types by passing all buffer and barrier types."""
        for buf_type in self.buffer_types:
            buf_type._flatten_ir_types(builder, out)
        for bar_type in self.barrier_types:
            bar_type._flatten_ir_types(builder, out)
        # Add counter type (int32 tensor)
        out.append(builder.get_int32_ty())

    def _unflatten_ir(self, handles, cursor):
        """Reconstruct ChannelReceiver from IR handles."""
        # First unflatten all buffers
        buffers = []
        for buf_type in self.buffer_types:
            buf, cursor = buf_type._unflatten_ir(handles, cursor)
            buffers.append(buf)

        # Then unflatten barriers
        empty_barriers, cursor = self.barrier_types[0]._unflatten_ir(handles, cursor)
        ready_barriers, cursor = self.barrier_types[1]._unflatten_ir(handles, cursor)

        # Unflatten counter (it's an int32 scalar)
        from triton.experimental.gluon.language import int32
        counter_type = ttgl.tensor([1], int32).type
        counter, cursor = counter_type._unflatten_ir(handles, cursor)

        # Reconstruct the receiver
        receiver = ChannelReceiver.__new__(ChannelReceiver)
        receiver.num_buffers = self.num_buffers
        receiver.shapes = self.shapes
        receiver.dtypes = self.dtypes
        receiver.layouts = self.layouts
        receiver.buffers = buffers
        receiver.empty_barriers = empty_barriers
        receiver.ready_barriers = ready_barriers
        receiver.counter = counter
        receiver.num_consumers = self.num_consumers
        receiver.consumer_id = self.consumer_id
        receiver.type = self

        return receiver, cursor


class ChannelReceiver(ttgl.base_value):
    """Consumer side of a channel.

    Maintains its own counter that is only modified by the receiver.
    """

    def __init__(self, channel: Channel, consumer_id: int):
        self.num_buffers = channel.num_buffers
        self.shapes = channel.shapes
        self.dtypes = channel.dtypes
        self.layouts = channel.layouts
        self.buffers = channel.buffers
        self.empty_barriers = channel.empty_barriers
        self.ready_barriers = channel.ready_barriers
        self.num_consumers = channel.num_consumers
        self.consumer_id = consumer_id

        # Create the type
        buffer_types = [buf.type for buf in self.buffers]
        barrier_types = [channel.empty_barriers.type, channel.ready_barriers.type]
        self.type = ChannelReceiverType(
            channel.num_buffers, channel.shapes, channel.dtypes, channel.layouts,
            buffer_types, barrier_types, channel.num_consumers, consumer_id
        )

        # Counter is an IR tensor that will be updated across calls
        self.counter = channel.semantic.to_tensor(ttgl.constexpr(0))

    def _flatten_ir(self, handles):
        """Flatten to IR by passing all buffers and barriers."""
        # Flatten all buffers
        for buf in self.buffers:
            buf._flatten_ir(handles)
        # Flatten barriers
        self.empty_barriers._flatten_ir(handles)
        self.ready_barriers._flatten_ir(handles)
        # Flatten counter
        self.counter._flatten_ir(handles)

    @ttgl.builtin
    def recv(self, _semantic=None):
        """Wait for and receive buffer(s).
        Returns (buffers_tuple, updated_receiver) to make state change explicit.
        """
        semantic = _semantic
        num_buffers = self.num_buffers
        counter = self.counter

        # Compute idx = (counter * num_consumers + consumer_id) % num_buffers
        num_bufs_tensor = semantic.to_tensor(ttgl.constexpr(num_buffers))
        num_cons_tensor = semantic.to_tensor(ttgl.constexpr(self.num_consumers))
        cons_id_tensor = semantic.to_tensor(ttgl.constexpr(self.consumer_id))

        temp = semantic.mul(counter, num_cons_tensor, sanitize_overflow=False)
        temp = semantic.add(temp, cons_id_tensor, sanitize_overflow=False)
        idx_tensor = semantic.mod(temp, num_bufs_tensor)

        # Compute phase = (counter // (num_buffers // num_consumers)) & 1 (wait for ready)
        # Each consumer revisits the same buffer slot every (num_buffers // num_consumers) increments
        one_tensor = semantic.to_tensor(ttgl.constexpr(1))
        buffers_per_consumer = num_buffers // self.num_consumers
        bufs_per_cons_tensor = semantic.to_tensor(ttgl.constexpr(buffers_per_consumer))
        div_result = semantic.floordiv(counter, bufs_per_cons_tensor)
        phase_tensor = semantic.and_(div_result, one_tensor)

        # Wait for ready barrier
        ready_idx = semantic.memdesc_index(self.ready_barriers, idx_tensor)
        pred_tensor = semantic.to_tensor(ttgl.constexpr(True))
        semantic.builder.create_mbarrier_wait(ready_idx.handle, phase_tensor.handle, pred_tensor.handle, [])

        # Index into each buffer and return
        result = []
        for buf in self.buffers:
            buf_idx = semantic.memdesc_index(buf, idx_tensor)
            result.append(buf_idx)

        # Increment counter (only receiver modifies this)
        new_counter = semantic.add(counter, one_tensor, sanitize_overflow=False)

        # Append channel metadata as the last two elements of the tuple
        # This ensures they survive compiler transformations
        result.append(idx_tensor)
        result.append(counter)

        # Create updated receiver with new counter
        updated_receiver = ChannelReceiver.__new__(ChannelReceiver)
        updated_receiver.num_buffers = self.num_buffers
        updated_receiver.shapes = self.shapes
        updated_receiver.dtypes = self.dtypes
        updated_receiver.layouts = self.layouts
        updated_receiver.buffers = self.buffers
        updated_receiver.empty_barriers = self.empty_barriers
        updated_receiver.ready_barriers = self.ready_barriers
        updated_receiver.num_consumers = self.num_consumers
        updated_receiver.consumer_id = self.consumer_id
        updated_receiver.counter = new_counter
        updated_receiver.type = self.type

        return ttgl.tuple([ttgl.tuple(result), updated_receiver])

    @ttgl.builtin
    def free(self, buffers, _semantic=None):
        """Signal that buffer(s) have been consumed.

        Args:
            buffers: The buffers returned from recv() - a tuple where the last two elements
                    are the channel index and counter
        """
        semantic = _semantic

        # Extract the channel metadata from the last two elements of the tuple
        # The tuple structure from recv() is: [buffer0, buffer1, ..., idx_tensor, counter]
        idx_tensor = buffers[-2]
        counter = buffers[-1]

        # Compute phase = (counter // (num_buffers // num_consumers)) & 1, then XOR with 1 (for empty barrier)
        num_bufs_tensor = semantic.to_tensor(ttgl.constexpr(self.num_buffers))
        one_tensor = semantic.to_tensor(ttgl.constexpr(1))
        buffers_per_consumer = self.num_buffers // self.num_consumers
        bufs_per_cons_tensor = semantic.to_tensor(ttgl.constexpr(buffers_per_consumer))
        div_result = semantic.floordiv(counter, bufs_per_cons_tensor)
        and_result = semantic.and_(div_result, one_tensor)
        phase_tensor = semantic.xor_(and_result, one_tensor)

        # Signal empty barrier
        empty_idx = semantic.memdesc_index(self.empty_barriers, idx_tensor)
        pred_tensor = semantic.to_tensor(ttgl.constexpr(True))
        semantic.builder.create_fence_async_shared(False)
        semantic.builder.create_mbarrier_arrive(empty_idx.handle, 1, pred_tensor.handle)


class _BarrierInsertingBuilder:
    """Wrapper around GluonOpBuilder that inserts barriers after operations."""

    def __init__(self, builder: GluonOpBuilder, semantic):
        self._builder = builder
        self._semantic = semantic
        self._in_region = False  # Track if we're inside a region (like reduction combine)

    def __getattr__(self, name):
        attr = getattr(self._builder, name)
        # If it's a create_* method (but not barrier/fence themselves or reduction ops), wrap it to insert barriers after
        skip_ops = ('create_barrier', 'create_fence_async_shared', 'create_reduce', 'create_scan')
        if callable(attr) and name.startswith('create_') and name not in skip_ops:
            def wrapped(*args, **kwargs):
                result = attr(*args, **kwargs)
                # Only insert barriers if we're not inside a region
                if not self._in_region:
                    self._builder.create_fence_async_shared(False)  # False = fence within CTA only
                    self._builder.create_barrier()
                return result
            return wrapped
        return attr


class GluonSemantic(TritonSemantic[TensorTy]):
    tensor = ttgl.tensor
    lang = ttgl

    builder: GluonOpBuilder

    def __init__(self, builder: GluonOpBuilder):
        if _GLUON_INSERT_BARRIERS:
            self.builder = _BarrierInsertingBuilder(builder, self)
        else:
            self.builder = builder

    def to_tensor(self, x, check_type=True):
        # Delegate to parent for normal handling
        return super().to_tensor(x, check_type)

    def _wrap_handle_infer_layout(self, handle, scalar_ty, shape):
        if shape == []:
            ty = scalar_ty
        else:
            ty = ttgl.distributed_type(scalar_ty, shape, self.builder.get_gluon_layout_from_tensor(handle))
        return self.tensor(handle, ty)

    def _wrap_tensor_infer_layout(self, tensor):
        return self._wrap_handle_infer_layout(tensor.handle, tensor.type.scalar, tensor.shape)

    def _broadcast_shapes(self, lhs_shape: List[int], rhs_shape: List[int]):
        if len(lhs_shape) != len(rhs_shape):
            raise ValueError(f"Cannot broadcast, rank mismatch: {lhs_shape}, {rhs_shape}")

        ret_shape = []
        for i, left in enumerate(lhs_shape):
            right = rhs_shape[i]
            if left == 1:
                ret_shape.append(right)
            elif (right == 1) or (right == left):
                ret_shape.append(left)
            else:
                raise ValueError("Cannot make_shape_compatible: incompatible dimensions "
                                 "at index " + str(i) + ": " + str(left) + " and " + str(right))
        return ret_shape

    def expand_dims(self, input: TensorTy, axis: int) -> TensorTy:
        dst_shape = [ttgl._unwrap_if_constexpr(x) for x in input.shape]
        dst_shape.insert(axis, 1)

        if axis < 0:
            axis += len(input.shape)

        _check(isinstance(input.type, ttgl.distributed_type),
               lambda: f"expected expand_dims input to be a distributed_type but got: {input.type!r}")
        layout = input.type.layout
        _check(isinstance(layout, (SliceLayout, AutoLayout)),
               lambda: f"expected expand_dims input to have a SliceLayout, but got: {layout}")
        _check(
            isinstance(layout, AutoLayout) or layout.dim == axis,
            lambda: f"expected expand_dims input layout to be sliced in axis {axis} but got {layout.dim}")

        handle = self.builder.create_expand_dims(input.handle, axis)
        return self._wrap_handle_infer_layout(handle, input.type.scalar, dst_shape)

    def join(self, a: TensorTy, b: TensorTy) -> TensorTy:
        a, b = self.broadcast_impl_value(a, b)
        _check(a.shape != [], lambda: "Cannot join scalars in gluon")
        value = super().join(a, b)
        return self._wrap_tensor_infer_layout(value)

    def split(self, a: TensorTy) -> Tuple[TensorTy, TensorTy]:
        lhs, rhs = super().split(a)
        return self._wrap_tensor_infer_layout(lhs), self._wrap_tensor_infer_layout(rhs)

    def permute(self, input: TensorTy, dims: Tuple[int]) -> TensorTy:
        value = super().permute(input, dims)
        return self._wrap_tensor_infer_layout(value)

    def broadcast_impl_shape(self, input: TensorTy, shape: Tuple[int]) -> TensorTy:
        _check(isinstance(input.type, ttgl.distributed_type),
               lambda: f"expected expand_dims input to be a distributed_type but got: {input.type!r}")
        src_shape = input.type.get_block_shapes()
        _check(len(src_shape) == len(shape), lambda: f"Cannot broadcast, rank mismatch: {src_shape}, {shape}")
        if shape == src_shape:
            return input
        for i, item in enumerate(src_shape):
            if shape[i] != item and item != 1:
                raise ValueError(f"Cannot broadcast, the expanded size of the tensor ({shape[i]})"
                                 f" must match the existing size ({item}) at non-singleton dimension"
                                 f" {i}: {src_shape}, {shape}")
        ret_ty = ttgl.distributed_type(input.type.scalar, shape, input.type.layout)
        handle = self.builder.create_broadcast(input.handle, ret_ty.to_ir(self.builder))
        return self.tensor(handle, ret_ty)

    def broadcast_impl_value(self, lhs: TensorTy, rhs: TensorTy) -> TensorTy:
        lhs_ty = lhs.type
        rhs_ty = rhs.type

        if not lhs_ty.is_block() or not rhs_ty.is_block():
            return super().broadcast_impl_value(lhs, rhs)

        _check(isinstance(lhs_ty, ttgl.distributed_type),
               lambda: f"expected broadcast left input to be a distributed_type but got: {lhs_ty!r}")
        _check(isinstance(rhs_ty, ttgl.distributed_type),
               lambda: f"expected broadcast right input to be a distributed_type but got: {rhs_ty!r}")

        lhs_shape = lhs_ty.get_block_shapes()
        rhs_shape = rhs_ty.get_block_shapes()
        ret_shape = self._broadcast_shapes(lhs_shape, rhs_shape)

        is_lhs_auto = isinstance(lhs_ty.layout, AutoLayout)
        is_rhs_auto = isinstance(rhs_ty.layout, AutoLayout)
        if is_lhs_auto and not is_rhs_auto:
            lhs = self.set_auto_layout(lhs, rhs_ty.layout)
        elif is_rhs_auto and not is_lhs_auto:
            rhs = self.set_auto_layout(rhs, lhs_ty.layout)
        elif lhs_ty.layout != rhs_ty.layout:
            raise ValueError(f"Layout mismatch in broadcast: {lhs_ty.layout} vs {rhs_ty.layout}")

        lhs = self.broadcast_impl_shape(lhs, ret_shape)
        rhs = self.broadcast_impl_shape(rhs, ret_shape)
        return lhs, rhs

    def arange(self, start, end, layout):
        shape = [end - start]
        if layout is None:
            layout = AutoLayout()
        ret_ty = ttgl.distributed_type(ttgl.int32, shape, layout)
        return super().arange(start, end, ret_ty=ret_ty)

    def reshape(self, input: TensorTy, dst_shape: List[int], can_reorder: bool):
        _check(not can_reorder, lambda: "can_reorder is not supported in gluon")
        value = super().reshape(input, dst_shape, can_reorder)
        return self._wrap_tensor_infer_layout(value)

    def splat(self, value, shape, layout):
        if len(shape) == 0:
            return value
        ret_ty = ttgl.distributed_type(value.dtype, shape, layout)
        handle = self.builder.create_splat(ret_ty.to_ir(self.builder), value.handle)
        return ttgl.tensor(handle, ret_ty)

    def full(self, shape, value, dtype, layout):
        scalar = self.make_scalar(value, dtype)
        if layout is None:
            layout = AutoLayout()
        return self.splat(scalar, shape, layout)

    def convert_layout(self, value, layout, assert_trivial=False):
        ty = value.type
        _check(isinstance(ty, ttgl.distributed_type),
               lambda: f"expected convert_layout input to be a distributed_type but got: {ty!r}")
        _check(isinstance(layout, ttgl.DistributedLayout),
               lambda: f"expected 'layout' to be a DistributedLayout but got {layout}")
        ret_ty = ttgl.distributed_type(ty.element_ty, ty.shape, layout)
        ret_ty_ir = ret_ty.to_ir(self.builder)
        if assert_trivial and not self.builder.is_convert_layout_trivial(ret_ty_ir, value.handle):
            raise TypeError(f"layout conversion from {ty.layout} to {layout} is not trivial.\n"
                            f"The linear layouts are:\n{self.to_linear_layout(ty.layout, ty.shape)}\n"
                            f"{self.to_linear_layout(layout, ty.shape)}")
        handle = self.builder.create_convert_layout(ret_ty_ir, value.handle)
        return ttgl.tensor(handle, ret_ty)

    def allocate_shared(self, element_ty, shape, layout, value):
        _check(isinstance(element_ty, ttgl.dtype), lambda: f"expected 'element_ty' to be a dtype but got {element_ty}")
        _check(_is_int_list(shape), lambda: f"all elements of 'shape' must be integers but got {shape}")
        _check(isinstance(layout, ttgl.SharedLayout),
               lambda: f"expected 'layout' to be a SharedLayout but got {layout}")
        ty = ttgl.shared_memory_descriptor_type(element_ty, shape, layout, shape)
        if value is not None:
            handle = self.builder.create_local_alloc(ty.to_ir(self.builder), value.handle)
        else:
            handle = self.builder.create_local_alloc(ty.to_ir(self.builder))
        return ttgl.shared_memory_descriptor(handle, element_ty, shape, layout, shape)

    def shared_load(self, mem_desc, layout):
        _check(isinstance(layout, ttgl.DistributedLayout),
               lambda: f"expected 'layout' to be a DistributedLayout but got {layout}")
        ret_ty = ttgl.distributed_type(mem_desc.dtype, mem_desc.shape, layout)
        handle = self.builder.create_local_load(ret_ty.to_ir(self.builder), mem_desc.handle)
        return ttgl.tensor(handle, ret_ty)

    def shared_store(self, mem_desc, value):
        _check(isinstance(value, ttgl.tensor), lambda: f"expected 'value' to be a tensor, but got a {type(value)}")
        _check(value.shape == mem_desc.shape,
               lambda: f"source shape {value.shape} and destination shape {mem_desc.shape} must match")
        _check(value.dtype == mem_desc.dtype,
               lambda: f"source dtype {value.dtype} and destination dtype {mem_desc.dtype} must match")
        self.builder.create_local_store(mem_desc.handle, value.handle)

    def shared_gather(self, mem_desc, indices, axis):
        _check(isinstance(indices, ttgl.tensor),
               lambda: f"expected 'indices' to be a tensor, but got a {type(indices)}")
        _check(isinstance(axis, int), lambda: f"expected 'axis' to be an int, but got a {type(axis)}")
        _check(
            len(indices.shape) == mem_desc.rank,
            lambda: f"indices rank must match memdesc rank: got {len(indices.shape)} and {mem_desc.rank}")
        _check(0 <= axis < mem_desc.rank, lambda: f"axis {axis} is out of bounds for memdesc rank {mem_desc.rank}")
        _check(indices.dtype.is_int(), lambda: f"indices must have integer dtype, got {indices.dtype}")

        ret_ty = ttgl.distributed_type(mem_desc.dtype, indices.shape, indices.type.layout)
        handle = self.builder.create_local_gather(ret_ty.to_ir(self.builder), mem_desc.handle, indices.handle, axis)
        return ttgl.tensor(handle, ret_ty)

    def shared_scatter(self, mem_desc, indices, axis, values, disjoint_group=None):
        _check(isinstance(indices, ttgl.tensor),
               lambda: f"expected 'indices' to be a tensor, but got a {type(indices)}")
        _check(isinstance(axis, int), lambda: f"expected 'axis' to be an int, but got a {type(axis)}")
        _check(isinstance(values, ttgl.tensor), lambda: f"expected 'values' to be a tensor, but got a {type(values)}")
        _check(
            len(indices.shape) == mem_desc.rank,
            lambda: f"indices rank must match memdesc rank: got {len(indices.shape)} and {mem_desc.rank}")
        _check(0 <= axis < mem_desc.rank, lambda: f"axis {axis} is out of bounds for memdesc rank {mem_desc.rank}")
        _check(indices.dtype.is_int(), lambda: f"indices must have integer dtype, got {indices.dtype}")
        _check(values.shape == indices.shape,
               lambda: f"values must have the same shape as indices: got {values.shape} and {indices.shape}")
        _check(values.type.layout == indices.type.layout, lambda: "values must have the same layout as indices")
        _check(
            values.dtype == mem_desc.dtype,
            lambda: f"values element type must match destination element type: got {values.dtype} and {mem_desc.dtype}")
        if disjoint_group is not None:
            _check(isinstance(disjoint_group, int) and disjoint_group >= 0,
                   lambda: f"expected 'disjoint_group' to be a non-negative int, but got {disjoint_group}")

        self.builder.create_local_scatter(mem_desc.handle, values.handle, indices.handle, axis, disjoint_group)

    def bank_conflicts(self, distr_ty, shared_ty):
        if not isinstance(distr_ty, ttgl.distributed_type):
            raise TypeError(
                f"bank_conflicts expects the register layout to be a distributed_type, got {type(distr_ty)}")

        if not isinstance(shared_ty, ttgl.shared_memory_descriptor_type):
            raise TypeError(
                f"bank_conflicts expects the shared layout to be a shared_memory_descriptor_type, got {type(shared_ty)}"
            )

        if distr_ty.shape != shared_ty.shape:
            raise ValueError(f"register shape {distr_ty.shape} and shared shape {shared_ty.shape} must match")
        if shared_ty.element_ty != distr_ty.element_ty:
            raise ValueError(
                f"mismatched dtypes between register ({distr_ty.element_ty}) and shared ({shared_ty.element_ty}) layouts"
            )
        if shared_ty.shape != shared_ty.alloc_shape[-len(shared_ty.shape):]:
            raise ValueError(
                f"bank_conflicts NYI for subslices. Got shape {shared_ty.shape} and alloc_shape {shared_ty.alloc_shape}"
            )

        reg_attr = distr_ty.layout._to_ir(self.builder)
        shared_attr = shared_ty.layout._to_ir(self.builder)
        return self.builder.get_shared_bank_conflicts(reg_attr, shared_attr, list(distr_ty.shape),
                                                      distr_ty.element_ty.primitive_bitwidth)

    def to_linear_layout(self, layout, shape):
        _check(isinstance(layout, (DistributedLayout, SharedLayout)),
               lambda: f"Expected a DistributedLayout or SharedLayout, got {type(layout)}")

        if not isinstance(shape, list):
            shape = list(shape)

        layout = ttgl._unwrap_if_constexpr(layout)

        if isinstance(layout, (AutoLayout, DistributedLinearLayout)):
            return ttgl.constexpr(layout)

        return ttgl.constexpr(self.builder.to_linear_layout(layout._to_ir(self.builder), shape))

    def shared_dealloc(self, mem_desc):
        self.builder.create_local_dealloc(mem_desc.handle)

    def set_auto_layout(self, value, layout):
        src_ty = value.type
        _check(isinstance(layout, DistributedLayout),
               lambda: f"set_auto_layout must set to a distributed layout but got {layout}")
        _check(isinstance(src_ty.layout, AutoLayout),
               lambda: f"set_auto_layout input must have auto layout but got {value.type.layout}")
        handle = self.builder.create_set_auto_layout(layout._to_ir(self.builder), value.handle)
        res_ty = ttgl.distributed_type(src_ty.element_ty, src_ty.shape, layout)
        return self.tensor(handle, res_ty)

    def memdesc_slice(self, mem_desc, start, length, dim):
        _check(isinstance(start, int), lambda: f"expected 'start' to be an int but got {start}")
        _check(isinstance(length, int), lambda: f"expected 'length' to be an int but got {length}")
        _check(isinstance(dim, int), lambda: f"expected 'dim' to be an int but got {dim}")
        offsets = [0] * mem_desc.rank
        offsets[dim] = start
        shape = list(mem_desc.shape)
        shape[dim] = length
        layout = mem_desc.layout
        ty = ttgl.shared_memory_descriptor_type(mem_desc.dtype, shape, layout, mem_desc.type.alloc_shape)
        builder = self.builder
        handle = builder.create_memdesc_subslice(ty.to_ir(builder), mem_desc.handle, offsets)
        return ttgl.shared_memory_descriptor(handle, **ty.__dict__)

    def memdesc_index(self, mem_desc, index):
        index = self.to_tensor(index)
        _check(index.type == ttgl.int32, lambda: f"expected 'index' to be int32 but got {index.type}")
        shape = mem_desc.shape[1:]
        index = self.to_tensor(index).handle
        layout = mem_desc.layout
        ty = ttgl.shared_memory_descriptor_type(mem_desc.dtype, shape, layout, shape)
        builder = self.builder
        handle = builder.create_memdesc_index(ty.to_ir(builder), mem_desc.handle, index)
        return ttgl.shared_memory_descriptor(handle, **ty.__dict__)

    def memdesc_trans(self, mem_desc, order):
        _check(_is_int_list(order), lambda: f"all elements of 'order' must be integers but got {order}")
        _check(
            len(order) == len(mem_desc.shape),
            lambda: f"source rank ({mem_desc.rank}) and order length ({len(order)}) must match")

        shape = [mem_desc.shape[i] for i in order]
        alloc_shape = mem_desc.type.alloc_shape
        new_alloc_shape = alloc_shape[:len(alloc_shape) - mem_desc.rank]
        new_alloc_shape += [alloc_shape[len(alloc_shape) - mem_desc.rank:][i] for i in order]

        handle = self.builder.create_memdesc_trans(mem_desc.handle, order)
        layout = self.builder.get_gluon_layout_from_memdesc(handle)
        return ttgl.shared_memory_descriptor(handle, element_ty=mem_desc.dtype, shape=shape,
                                             alloc_shape=new_alloc_shape, layout=layout)

    def memdesc_reshape(self, mem_desc, shape):
        _check(_is_int_list(shape), lambda: f"all elements of 'shape' must be integers but got {shape}")
        _check(
            math.prod(shape) == math.prod(mem_desc.shape),
            lambda: (f"memdesc_reshape total elements mismatch: "
                     f"{mem_desc.shape} -> {shape}"),
        )

        handle = self.builder.create_memdesc_reshape(mem_desc.handle, shape)
        layout = self.builder.get_gluon_layout_from_memdesc(handle)
        alloc_shape = mem_desc.type.alloc_shape
        prefix_len = len(alloc_shape) - mem_desc.rank
        new_alloc_shape = alloc_shape[:prefix_len] + list(shape)

        return ttgl.shared_memory_descriptor(
            handle,
            element_ty=mem_desc.dtype,
            shape=shape,
            alloc_shape=new_alloc_shape,
            layout=layout,
        )

    def memdesc_reinterpret(self, mem_desc, dtype, shape, layout):
        _check(isinstance(dtype, ttgl.dtype), lambda: f"expected 'dtype' to be a dtype but got {dtype}")
        _check(_is_int_list(shape), lambda: f"all elements of 'shape' must be integers but got {shape}")
        _check(isinstance(layout, ttgl.SharedLayout),
               lambda: f"expected 'layout' to be a SharedLayout but got {layout}")
        ty = ttgl.shared_memory_descriptor_type(dtype, shape, layout, shape)
        handle = self.builder.create_memdesc_reinterpret(ty.to_ir(self.builder), mem_desc.handle)
        return ttgl.shared_memory_descriptor(handle, **ty.__dict__)

    def wrap_tensor(self, x, scalar_ty, ret_shape, layout):
        if ret_shape:
            res_ty = ttgl.distributed_type(scalar_ty, ret_shape, layout)
        else:
            res_ty = scalar_ty
        return self.tensor(x, res_ty)

    @staticmethod
    def _check_same_layout(xs):
        for x in xs:
            _check(isinstance(x.type, ttgl.distributed_type), lambda: f"expected distributed_type but got: {x.type!r}")
        layouts = [x.type.layout for x in xs]
        l0 = layouts[0]
        _check(all(l == l0 for l in layouts[1:]),
               lambda: f"Expected inputs to have matching layouts, but got: {layouts}")

    def associative_scan(self, inputs: Sequence[TensorTy], axis: int, region_builder_fn,
                         reverse: bool) -> Tuple[TensorTy, ...]:
        shape = inputs[0].type.shape
        rank = len(shape)

        assert -rank <= axis < rank, f"scan axis {axis} must be < inputs rank ({rank})"

        if axis < 0:
            axis += rank

        for t in inputs:
            assert t.type.shape == shape, "all scan inputs must have the same shape"

        scan_op = self.builder.create_scan([t.handle for t in inputs], axis, reverse)
        region_builder_fn(scan_op)
        assert scan_op.verify()

        return tuple(
            self._wrap_handle_infer_layout(scan_op.get_result(i), inputs[i].type.scalar, shape)
            for i in range(len(inputs)))

    def reduction(self, inputs: Sequence[TensorTy], axis: int, region_builder_fn) -> Tuple[TensorTy, ...]:
        if axis is None:
            inputs = tuple(self.reshape(t, [t.numel.value], can_reorder=False) for t in inputs)
            axis = 0
        # get result shape
        shape = inputs[0].type.shape
        rank = len(shape)
        _check(0 <= axis < rank, lambda: f"expected reduction axis to be in the range [0, {rank}) but got {axis}")
        self._check_same_layout(inputs)
        ret_shape = [s for i, s in enumerate(shape) if i != axis]
        assert all(t.type.shape == shape for t in inputs), "all reduction inputs must have the same shape"

        reduce_op = self.builder.create_reduce([t.handle for t in inputs], axis)

        # Mark that we're inside a region to prevent barrier insertion
        if isinstance(self.builder, _BarrierInsertingBuilder):
            old_in_region = self.builder._in_region
            self.builder._in_region = True

        region_builder_fn(reduce_op)

        # Restore the flag
        if isinstance(self.builder, _BarrierInsertingBuilder):
            self.builder._in_region = old_in_region

        assert reduce_op.verify()

        return tuple(
            self._wrap_handle_infer_layout(reduce_op.get_result(i), inputs[i].type.scalar, ret_shape)
            for i in range(len(inputs)))

    def histogram(self, input: TensorTy, num_bins: int, mask: TensorTy, layout) -> TensorTy:
        _check(len(input.shape) == 1, lambda: "histogram only supports 1D input")
        _check(input.dtype.is_int(), lambda: "histogram only supports integer input")
        _check(layout is not None, lambda: "histogram requires a destination layout")
        if mask is not None:
            mask, input = self.broadcast_impl_value(mask, input)
            _check(mask.type.scalar.is_bool(), lambda: "Mask must have boolean scalar type")
            mask = mask.handle
        layout_attr = layout._to_ir(self.builder)
        handle = self.builder.create_histogram(input.handle, num_bins, mask, layout_attr)
        return self.wrap_tensor(handle, ttgl.int32, [num_bins], layout)

    def gather(self, src: TensorTy, index: TensorTy, axis: int) -> TensorTy:
        _check(isinstance(src.type, ttgl.distributed_type), lambda: f"expected distributed_type but got: {src.type!r}")
        _check(isinstance(index.type, ttgl.distributed_type),
               lambda: f"expected distributed_type but got: {index.type!r}")
        _check(index.type.scalar.is_int(), lambda: f"expected integer scalar type but got: {index.type.scalar!r}")

        rank = len(src.type.shape)
        _check(len(index.type.shape) == rank, lambda: "source and index tensors must have the same rank")
        _check(-rank <= axis < rank, lambda: f"gather axis {axis} must be < source rank ({rank})")
        if axis < 0:
            axis += rank

        for d in range(rank):
            if d == axis:
                continue
            _check(
                index.type.shape[d] == src.type.shape[d],
                lambda: f"index dim {axis} must match the corresponding source dim",
            )
        gather = self.builder.create_gather(src.handle, index.handle, axis)
        return self.wrap_tensor(gather, src.type.scalar, index.type.shape, index.type.layout)

    def warp_specialize(self, functions_and_args, worker_num_warps: Sequence[int], worker_num_regs: Sequence[int],
                        generator):
        for _, args in functions_and_args:
            _check(isinstance(args, (tuple, ttgl.tuple)),
                   lambda: f"function arguments must be a tuple of arguments, but got {type(args)}")

        assert len(functions_and_args) >= 1, "expected at least one function for the default partition"
        default_partition, default_args = functions_and_args[0]
        num_partitions = len(functions_and_args) - 1
        workers = functions_and_args[1:]

        assert num_partitions == len(
            worker_num_warps
        ), f"warp specialize got {num_partitions} partitions but {len(worker_num_warps)} warp counts"
        assert num_partitions == len(
            worker_num_regs
        ), f"warp specialize got {num_partitions} partitions but {len(worker_num_regs)} register counts"

        builder = self.builder
        insert_pt = builder.get_insertion_point()

        # Emit the default partition to get the result types.
        default_block = builder.new_block()
        builder.set_insertion_point_to_start(default_block)
        default_results = generator.call_JitFunction(default_partition, default_args, kwargs={})
        mlir_results = []
        if default_results is not None:
            mlir_results = flatten_values_to_ir(default_results)
        builder.create_warp_yield(mlir_results)
        result_types = [r.get_type() for r in mlir_results]

        # Create the warp specialize op.
        worker_args = [flatten_values_to_ir(args) for _, args in workers]
        mlir_args = sum(worker_args, [])
        builder.restore_insertion_point(insert_pt)
        ws_op = builder.create_warp_specialize(result_types, mlir_args, worker_num_warps)
        ws_op.get_default_region().push_back(default_block)
        ws_op.set_requested_registers(worker_num_regs)

        # Emit the partition regions.
        builder.create_block_with_parent(ws_op.get_partition_op_holder(), [])
        partitions_op = builder.create_warp_specialize_partitions(num_partitions)
        arg_types = [arg.get_type() for arg in mlir_args]
        arg_it = 0
        for i, (func, args) in enumerate(workers):
            caller_context = GluonCallerContext(num_warps=worker_num_warps[i])
            block = builder.create_block_with_parent(partitions_op.get_region(i), arg_types)
            mlir_args = worker_args[i]
            block_args = [block.get_argument(arg_it + j) for j in range(len(mlir_args))]
            block_args = unflatten_ir_values(block_args, [arg.type for arg in args])
            generator.call_JitFunction(func, block_args, kwargs={}, caller_context=caller_context)
            builder.create_warp_return()
            arg_it += len(mlir_args)

        builder.set_insertion_point_after(ws_op.get_operation())
        mlir_results = [ws_op.get_result(i) for i in range(len(result_types))]
        if default_results is None:
            return
        return tuple(unflatten_ir_values(mlir_results, [r.type for r in default_results]))


    def create_channel(self, num_buffers, shapes, dtypes, layouts, num_producers=1, num_consumers=1):
        """Create a multi-buffered channel for producer-consumer communication.

        Args:
            num_buffers: Number of buffers in the circular queue
            shapes: List of tensor shapes (one per tensor in the bundle)
            dtypes: List of dtypes (one per tensor)
            layouts: List of SharedLayouts (one per tensor)
            num_producers: Number of producers (default 1)
            num_consumers: Number of consumers (default 1)

        Returns:
            Tuple of (senders, receivers) where:
            - senders is a single sender if num_producers==1, else a list of senders
            - receivers is a single receiver if num_consumers==1, else a list of receivers
        """
        channel = Channel(self, num_buffers, shapes, dtypes, layouts, num_producers, num_consumers)

        # Create senders
        if num_producers == 1:
            senders = channel.sender(0)
        else:
            senders = ttgl.tuple([channel.sender(i) for i in range(num_producers)])

        # Create receivers
        if num_consumers == 1:
            receivers = channel.receiver(0)
        else:
            receivers = ttgl.tuple([channel.receiver(i) for i in range(num_consumers)])

        return ttgl.tuple([senders, receivers])

    def num_warps(self, generator):
        if generator.caller_context is not None:
            assert isinstance(generator.caller_context, GluonCallerContext)
            return ttgl.constexpr(generator.caller_context.num_warps)
        return ttgl.constexpr(self.builder.options.num_warps)
