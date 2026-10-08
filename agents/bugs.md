# Bugs

## Large local aggregates exhaust the native stack

Every scalar narrower than 64 bits occupies four bytes, so `[1000000]u8` takes
4 MB. Each aggregate load holds its own copy in storage allocated at function
entry, and an array literal is built in a temporary local before it is copied
into its binding, so one declaration can hold several copies at once.
Exhausting the stack ends the process with a bare SIGSEGV rather than a trap
diagnostic.
