# Upstream pyrefly work

Issues and fixes vision3d would like in [facebook/pyrefly](https://github.com/facebook/pyrefly),
found while adding tensor shape annotations to this codebase.

Measured against `pyrefly==1.2.0.dev3`, `pyrefly-torch-stubs==1.2.0.dev3`,
`pyrefly-shape-extensions==1.2.0.dev3`, `torch==2.12.1`.

Each item is marked **verified** (the fix or the behaviour was reproduced
locally) or **inferred** (read from the pyrefly source, not compiled and run).

Reproduce the raw diagnostics with:

```bash
PS=path/to/pyrefly/tensor-shapes
uvx pyrefly@1.2.0.dev3 check --config /dev/null --python-version 3.12 \
  --search-path src \
  --search-path $PS/pyrefly-torch-stubs \
  --search-path $PS/pyrefly-shape-extensions \
  --python-interpreter-path .venv/bin/python 'src/vision3d/**/*.py'
```

Ranked by value per line of upstream change.

---

## 1. `@torch.no_grad()` erases the decorated signature

- Kind: bug
- Size: 1 line
- Status: fix verified, not filed

`torch-stubs/__init__.pyi:2484` types the decorator as returning `Any`:

```python
class no_grad:
    def __call__(self, func) -> Any: ...  # For decorator usage
```

so every decorated function loses its whole signature — return type *and*
argument checking:

```python
@torch.no_grad()
def box3d_iou[N: IntVar, M: IntVar, K: IntVar](
    boxes1: Tensor[[N, K]], boxes2: Tensor[[M, K]], format: BoundingBox3DFormat
) -> Tensor[[N, M]]: ...

box3d_iou(b1, b2, fmt)   # revealed: Unknown   (want Tensor[[N, M]])
box3d_iou(b1, scores, fmt)  # mismatched shapes: silently accepted
```

Using `with torch.no_grad():` instead keeps the signature, which shows the
body is fine and only the decorator path is affected.

### Fix

```python
def __call__[F](self, func: F) -> F: ...
```

Verified by patching a local copy of the stubs:

| | before | after |
| --- | --- | --- |
| `box3d_iou(...)` | `Unknown` | `Tensor[[N, M]]` |
| `nms_3d(...)` | `Unknown` | `Tensor[[int]]` |
| `voxelize(...)` | `Unknown` | `tuple` of 3 shaped tensors |
| bad argument | accepted | `ERROR ... not assignable to parameter boxes2` |

Affects 4 public ops here. The same pattern probably applies to the other
`_DecoratorContextManager` subclasses (`enable_grad`, `inference_mode`).

---

## 2. Tensor subclasses degrade to gradual ("T1")

- Kind: bug (soundness)
- Size: ~10 lines
- Status: inferred, not filed

Adopting the shape stubs silently disables *all* checking on `torch.Tensor`
subclasses. With them installed, a `PointCloud3D` parameter accepts anything:

```python
def only_pc(x: PointCloud3D) -> None: ...

only_pc(some_tensor)   # no error
only_pc(some_boxes)    # no error - different tv_tensor subclass
only_pc(42)            # no error
only_pc("hello")       # no error
```

Same file, same checker, only the interpreter differs:

| | result |
| --- | --- |
| pyrefly **with** shape stubs | `0 errors` |
| pyrefly **without** shape stubs | `ERROR ... not assignable to PointCloud3D` |

Inference dies at the same boundary — every expression touching a subclass is
`Unknown`, which then propagates:

| expression | with stubs | without |
| --- | --- | --- |
| `pc[:, :3]` | `Unknown` | `Tensor` |
| `pc.shape` | `Unknown` | `Size \| list[int] \| tuple[int, ...]` |
| `pc.unsqueeze(0)` | `Unknown` | `Tensor` |

**Cause.** `torch.Tensor` resolves to `Type::ShapedArray`, not a nominal
`ClassType`, so `class TVTensor(torch.Tensor)` fails base-class resolution and
the subclass falls back to gradual. Declaring one directly says so:

```text
ERROR Invalid base class: `Tensor` [invalid-inheritance]
```

**Fix.** `ShapedArrayType` already carries the nominal class:

```rust
pub struct ShapedArrayType {
    pub base_class: ClassType,   // e.g. torch.Tensor
    shape: ShapedArrayShapeStorage,
    pub syntax: ShapedArraySyntax,
}
```

`parse_base_class_type` (`pyrefly/lib/alt/class/class_metadata.rs:1452`)
matches `ClassType`, `Tuple`, `TypedDict`, … but has no `ShapedArray` arm, so
it falls through to `InvalidType`. Adding one that unwraps to
`shaped.base_class`, mirroring the existing `ClassType` arm, should restore the
hierarchy.

Subclasses would not carry shapes — that is item 3 — but they would be real
classes again.

**Impact here:** 309 tv_tensor annotations, all currently holes that swallow
both type errors and inference.

```text
transforms  200    tensors  38    metrics  5
datasets     53    viz      13    ops      0
```

**Risk:** the shape machinery may assume `ShapedArray` never appears in an MRO,
so a patched build could surface assertions elsewhere. Treat ~10 lines as the
optimistic floor.

---

## 3. Allow subtyping between shaped arrays

- Kind: feature
- Size: ~50-150 lines
- Status: inferred, not filed (upstream invites the issue)

This is what would let vision3d's tv_tensors carry real shapes — e.g.
`PointCloud3D` as `[N, 3+C]` — so dimensions thread across call boundaries
instead of going gradual at every tv_tensor argument.

The `@shape_extensions.shaped_array(shape="Shape")` decorator already produces
a working shaped class; the only blocker is that it is not assignable to
`Tensor[[...]]`:

```python
@shape_extensions.shaped_array(shape="Shape")
class PC[Shape: IntTuple](TVTensor):
    shape: Shape

def f[N: IntVar](x: Tensor[[N, 3]]) -> None: ...
def use[N: IntVar](pc: PC[[N, 3]]) -> None:
    f(pc)
```

```text
ERROR Argument `PC[[N, 3]]` is not assignable to parameter `x`
  with type `Tensor[[@_, 3]]`
  Pyrefly does not support subtyping relationships between shaped arrays
  `PC` and `torch.Tensor` at this time. If you need this, consider filing an issue.
```

The machinery exists and is deliberately gated —
`pyrefly/lib/solver/subset.rs:2830`:

```rust
let got_base = self.shape_erased_base_class(got, shape_param)?;
let want_base = self.shape_erased_base_class(want, want_param)?;
let same_class = got_base.class_object() == want_base.class_object();
self.is_subset_eq(&got_base.to_type(), &want_base.to_type())?;   // passes

// We do not (yet) support subtyping for shaped arrays given that
// there's no known need and it would complicate the shape param
// analysis. We need to catch this explicitly since the ClassType would
// be assignable.
if !same_class {
    return Err(SubsetError::ShapedArraySubtyping(...));
}
```

The base-class subset check above the guard already succeeds, and the shape
comparison below it (`bind_tensor_dimensions`) is class-agnostic.

**Fix.** Relax `same_class` to "nominal subclass of", and map a subclass's own
shape parameter onto the base's through the MRO — the `want_param !=
shape_param` check just below currently assumes the same class. That mapping is
the real work the comment alludes to, but it is local to this file.

**We are the "known need."** A torchvision-style library where every public
type (`BoundingBoxes3D`, `PointCloud3D`, `CameraImages`, `CameraIntrinsics`,
`CameraExtrinsics`) is a `Tensor` subclass is exactly the missing use case.

**Caveat for us, not upstream:** `@shaped_array` runs at import, so adopting it
makes `shape_extensions` a hard runtime dependency of vision3d.

**Depends on item 2** — see [Sequencing](#sequencing).

---

## 4. Wrong-shape bugs

- Kind: bugs
- Status: verified, not filed

These return an incorrect shape rather than an unknown one, which is worse than
no tracking — downstream code is checked against a wrong answer.

### `Tensor.T` is modelled as identity

```python
x: Tensor[[4, 5]]
x.T                    # revealed: Tensor[[4, 5]]   want [5, 4]
y: Tensor[[5, 4]] = x.T   # ERROR: rejects the correct annotation
x.transpose(0, 1)      # revealed: Tensor[[5, 4]]   correct
```

Worked around in `ops/_project.py` and `datasets/kitti.py` by spelling the
transposes `transpose(0, 1)`.

### `Tensor.expand` truncates when adding leading dimensions

Correct when rank is preserved, wrong when rank grows:

| expression | pyrefly | actual |
| --- | --- | --- |
| `Tensor[[4,4]].expand(2,4,4)` | `Tensor[[2, 4]]` | `[2,4,4]` |
| `Tensor[[3]].expand(7,3)` | `Tensor[[7]]` | `[7,3]` |
| `Tensor[[3]].expand(2,5,3)` | `Tensor[[2]]` | `[2,5,3]` |
| `Tensor[[1,5]].expand(9,5)` | `Tensor[[9, 5]]` | correct |
| `Tensor[[1,5]].expand(-1,5)` | `Tensor[[1, 5]]` | correct |

### `.item()` rejects single-element non-0-d tensors

```text
ERROR item() only works on 0-dimensional tensors, got 2D tensor
```

PyTorch allows `.item()` on any tensor with exactly one element; verified at
runtime that `torch.tensor([[0.5]]).item()` returns `0.5`. 12 occurrences here,
all in tests calling `.item()` on a `[1, 1]` IoU result.

### Matmul does not check inner dimensions

```python
a: Tensor[[4, 4]]; b: Tensor[[3, 3]]
a @ b     # revealed: Tensor[[4, 3]] - no error, but invalid at runtime
```

Outer dimensions are taken without verifying the inner ones agree.

---

## 5. Missing APIs

- Kind: stub coverage
- Status: verified
- Related: [#3380][i3380] (open)

The shape stubs shadow torch's own, so anything they do not model reads as
missing. This overlaps #3380, whose reporter hit a largely disjoint set
(`torch.square`, `torch.equal`, `torch.load`, `nn.RNN`, `nn.Transformer`,
`F.one_hot`) — the gap is broad rather than a fixed list.

Everything below is used by vision3d today.

### Module-level (18)

```text
torch.allclose      torch.as_tensor     torch.cdist        torch._check
torch.compile       torch.from_numpy    torch.is_floating_point
torch.isnan         torch.manual_seed   torch.nan_to_num   torch.nonzero
torch.promote_types torch.quantile      torch.randperm     torch.searchsorted
torch.uint8         torch.unique        torch.linalg.norm
```

Also `torch.nan` and `torch.Generator`.

### Tensor methods (13)

```text
all   allclose   any    argsort   as_subclass   clamp_min   data_ptr
fill_diagonal_   flip   __invert__   is_floating_point   neg_   numpy
```

Also `Tensor.mT` — notable because it is PyTorch's recommended replacement for
`.T` on rank ≠ 2, so item 4's workaround cannot use it.

`any`/`all` may be blocked by
[#3323][i3323] (method names shadowing builtins), which the stub already warns about.

### Signature gaps

```text
torch.eye(dtype=, device=)      torch.full(dtype=, device=)
torch.zeros_like(dtype=)        torch.where(condition)      # 1-arg form
Tensor.new_empty(tuple)         # tuple-of-sizes overload
```

### Operators

```text
Tensor.__and__ / __iand__ / __invert__    # bool-mask combination, 13 sites
Tensor.__iter__                            # zip(tensor) fails: not Iterable
torch.device as a context manager          # `with torch.device(...)`: 8 sites
```

### Shape tracking through containers

 `cat`/`stack` propagate shapes only
through the tuple form:

```python
torch.cat((a, b), dim=0)   # revealed: Tensor[[8, 5]]
torch.cat([a, b], dim=0)   # revealed: Tensor[tuple[Unknown, ...]]
```

---

## 6. Errors inside the stub files

- Kind: bug
- Status: verified
- Related: [#3863][i3863] (open)

Vendoring `torch-stubs` into a project and putting it on `search-path` adds 451
diagnostics *from the `.pyi` files themselves* (254 in `torch-stubs/`, 197 in
`torch-stubs/nn/`), because they are then inside the project tree. Needs
`project-excludes`, and matches #3863.

---

## Sequencing

Land item 1 whenever — it is a self-contained stub change.

Items 2 and 3 should go up as a **stacked pair, 2 then 3**. They touch
different files (`alt/class/class_metadata.rs` and `solver/subset.rs`) so the
patches do not conflict, but item 3 is not honestly testable without item 2.

The guard in item 3 sits directly below a base-class check:

```rust
self.is_subset_eq(&got_base.to_type(), &want_base.to_type())?;
```

For `PC[[N, 3]] -> Tensor[[N, 3]]` that asks whether `PC` relates to
`torch.Tensor`. Today that relation holds only vacuously. `is_any()` classifies
an unresolvable base as *any*:

```rust
BaseClassParseResult::InvalidBase(..)
| BaseClassParseResult::InvalidExpr(..)
| BaseClassParseResult::InvalidType(..)     // "Invalid base class: Tensor"
| BaseClassParseResult::AnyType => true,
```

so `TVTensor`'s unresolvable `torch.Tensor` base makes the whole class gradual.
Confirmed: `wants_tvtensor("a string")` reports 0 errors. Every
`TVTensor <: torch.Tensor` result is therefore satisfied by gradualness rather
than a real MRO edge.

Consequences:

- Relaxing the guard alone would likely *appear* to work, but only because the
  base check passes vacuously. That rests on the bug, and could change once
  item 2 lands.
- The natural test case for item 3 is a shape-parameterised
  `class MyTensor(torch.Tensor)`, which is still `Invalid base class: Tensor`
  until item 2 lands. Testing via `@shaped_array` on a class whose `Tensor`
  relationship is fake does not exercise the relation the patch claims to
  enable.

Item 2 first therefore gives item 3 both a sound hierarchy and a real test.

**Confidence.** The dependency is inferred, not proven — the gradual
degradation masks the MRO from outside the compiler, so it cannot be
distinguished from a genuine edge by probing alone. The `is_any()` source above
is the strongest evidence. To confirm, patch item 2 locally and re-run the
item 3 repro: if the guard error still fires against a now-sound hierarchy,
item 3 is the only remaining blocker.

---

## Related open issues

| Issue | Title | Relevance |
| --- | --- | --- |
| [#3380][i3380] | torch stub missing many methods | item 5; confirmed shapes-specific |
| [#3311][i3311] | Bundled fixtures without copy-paste | deferred as a design question |
| [#3863][i3863] | torch-shape stub has type error | item 6 |
| [#3323][i3323] | Method names shadow builtins in torch stubs | `any`/`all` in item 5 |
| [#3985][i3985] | pyrefly-torch-stubs cause high CPU usage | know before adoption |

Items 1–4 have no matching issue.

---

## Notes for contributing

`TENSOR_SHAPES_CONTRIBUTING.md` states most external contributions should be
stub-only, and that stub and DSL changes do not require touching the Rust
internals. That covers items 1, 4 and 5. Items 2 and 3 are kernel work.

There is no user-side mechanism for extending the stubs. All of these were
tested and shadow rather than merge:

- PEP 561 partial stubs (`partial` in `py.typed`) — unsupported
- a submodule-only overlay — claims the whole package root
- re-export composition — blocked by upstream's `__all__ = ["Tensor"]`
- `site-package-path`, the documented pyright `stubPath` equivalent — shadows identically

So local additions mean vendoring the whole `torch-stubs` tree, which is why
upstreaming is preferred.

[i3311]: https://github.com/facebook/pyrefly/issues/3311
[i3323]: https://github.com/facebook/pyrefly/issues/3323
[i3380]: https://github.com/facebook/pyrefly/issues/3380
[i3863]: https://github.com/facebook/pyrefly/issues/3863
[i3985]: https://github.com/facebook/pyrefly/issues/3985
