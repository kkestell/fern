# Bugs

- Small functions over structs are slow

  Every struct parameter is copied into a stack slot on entry, and every struct
  local or field read into a local is copied with `blit`, so its fields are
  loaded from memory rather than kept in registers. No call is inlined. In
  `examples/raytrace.fern`, moving the box intersection from one written-out
  function into `clip`, `local_point`, and `local_direction` raised the render
  from 12.2 to 16.3 CPU seconds.
