# Bugs

## Stack overflow is unchecked

Compiled code neither probes the stack nor checks for overflow. Each function
allocates its whole frame at entry with one stack-pointer adjustment, so deep
recursion or a large frame runs past the end of the stack. Usually the process
ends with a bare SIGSEGV rather than a trap diagnostic. A frame larger than the
guard page can move the stack pointer past the guard page entirely, so the
frame's first access may land in other mapped memory instead of faulting. That
makes stack overflow memory-unsafe.

Large local aggregates make large frames. Every scalar narrower than 64 bits
occupies four bytes, so `[1000000]u8` takes 4 MB. Each aggregate load holds its
own copy in storage allocated at function entry, and an array literal is built
in a temporary local before it is copied into its binding, so one declaration
can hold several copies at once.
