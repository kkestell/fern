#!/usr/bin/env bash
# Codebase health report for Fern.
#
# Every function must stay at or under a cognitive complexity of 15, as
# measured by rust-code-analysis-cli. Test code is not measured: `#[cfg(test)]`
# modules in src/ and the tests/ directory are both skipped.
#
# Usage:
#   scripts/code-health.sh check          fail if any function is over the budget
#   scripts/code-health.sh report [OUT]   write the full per-function report

set -euo pipefail

TOOL_VERSION="0.0.25"
BUDGET=15

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

die() {
  echo "code-health: $*" >&2
  exit 1
}

command -v jq >/dev/null || die "jq is not installed"
command -v rust-code-analysis-cli >/dev/null ||
  die "rust-code-analysis-cli is not installed; run: cargo install rust-code-analysis-cli --version $TOOL_VERSION --locked"
installed=$(rust-code-analysis-cli --version | awk '{print $2}')
[ "$installed" = "$TOOL_VERSION" ] ||
  die "rust-code-analysis-cli $installed is installed, but this report is pinned to $TOOL_VERSION"

# Print "file<TAB>first<TAB>last" for each `#[cfg(test)] mod` block in src/.
# The spans come from the parser, because a closing brace in column one can
# also belong to a Fern program inside a test's raw string.
test_modules() {
  local file
  for file in src/*.rs; do
    rust-code-analysis-cli --paths "$file" --find mod_item |
      sed 's/\x1b\[[0-9;]*m//g' |
      sed -n 's/.*from (\([0-9]*\), [0-9]*) to (\([0-9]*\),.*/\1 \2/p' |
      while read -r first last; do
        sed -n "$((first - 1))p" "$file" | grep -q '^#\[cfg(test)\]$' &&
          printf '%s\t%d\t%d\n' "$file" "$first" "$last"
      done
  done
}

# Write the report to $1.
report() {
  local raw
  raw=$(mktemp -d)
  trap 'rm -rf "$raw"' RETURN

  # Passing "." produces no reports at all, so name the source directory.
  rust-code-analysis-cli --paths src --metrics \
    --output-format json --pr --output "$raw" --include '**/*.rs'

  local files
  files=$(find "$raw" -name '*.json' | sort)
  [ -n "$files" ] || die "rust-code-analysis-cli produced no reports"

  local tests
  tests=$(test_modules | jq -R -s 'split("\n") | map(select(length > 0) | split("\t")
    | {file: .[0], first: (.[1] | tonumber), last: (.[2] | tonumber)})')

  # shellcheck disable=SC2086
  jq -s --arg root "$repo_root/" --argjson budget "$BUDGET" --argjson tests "$tests" '
    def functions:
      .spaces[]? | recurse(.spaces[]?) | select(.kind == "function");

    [ .[]
      | (.name | sub("^" + $root; "")) as $file
      | functions
      | { file: $file, name, line: .start_line } as $header
      | select([ $tests[] | select(.file == $header.file
          and .first <= $header.line and $header.line <= .last) ] | length == 0)
      | $header + {
          sloc: (.metrics.loc.sloc | round),
          cognitive: (.metrics.cognitive.sum | round),
          cyclomatic: (.metrics.cyclomatic.sum | round),
        }
    ]
    | sort_by(-.cognitive, .file, .line) as $functions
    | {
        tool: "rust-code-analysis-cli " + $ARGS.named.version,
        budget: $budget,
        totals: {
          sloc: ([$functions[].sloc] | add),
          functions: ($functions | length),
          over_budget: ([$functions[] | select(.cognitive > $budget)] | length),
          max_cognitive: ([$functions[].cognitive] | max),
        },
        functions: $functions,
      }
  ' --arg version "$TOOL_VERSION" $files > "$1"
}

case "${1:-}" in
report)
  out=${2:-target/code-health/report.json}
  mkdir -p "$(dirname "$out")"
  report "$out"
  echo "wrote $out"
  ;;
check)
  mkdir -p target/code-health
  report target/code-health/report.json
  jq -r '
    "\(.totals.functions) functions, \(.totals.sloc) sloc, budget \(.budget) cognitive complexity",
    "",
    (if .totals.over_budget == 0 then "Every function is within budget."
     else "\(.totals.over_budget) functions are over budget:" end),
    (.budget as $budget | .functions[] | select(.cognitive > $budget)
      | "  \(.file):\(.line)  \(.name)  cognitive \(.cognitive)")
  ' target/code-health/report.json
  jq -e '.totals.over_budget == 0' target/code-health/report.json >/dev/null
  ;;
*)
  sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
  ;;
esac
