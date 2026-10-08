# Todo

## Build returned aggregate literals in the result storage

A `return` whose value is an array or struct literal builds the literal in a
temporary local, copies it into the load's storage, and copies that into the
caller's result storage. Build it in the result storage directly.

## Give a returned local the result storage

A function that returns the same aggregate local on every `return` copies that
local into the caller's result storage. Allocate the local in the result
storage instead, so the `return` needs no copy.

## Initialize bindings from calls without a copy

`const b = make();` writes the result into the call's own storage and then
copies it into `b`. Pass `b`'s storage as the result address instead.
