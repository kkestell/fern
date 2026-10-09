# Bugs

- Small functions over structs are slow

  No call is inlined, and an aggregate result returns through memory the
  caller provides. An aggregate with more than eight scalars and pointers,
  such as `Hit` in `examples/raytrace.fern`, is still copied with `blit` or
  `memcpy`, which keeps it in a stack slot. In `examples/raytrace.fern`,
  moving the box intersection from one written-out function into `clip`,
  `local_point`, and `local_direction` raised the render from 12.2 to 16.3 CPU
  seconds; copying small aggregates field by field brought it to 13.4.
