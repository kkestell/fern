def format_header(label: str, width: int = 80, fill_char: str = "═") -> str:
    bold_white = "\033[1;97m"
    reset = "\033[0m"
    padding = width - len(label) - 4
    header = f"{fill_char} {label} {fill_char * padding}"
    return f"\n{bold_white}{header}{reset}"
