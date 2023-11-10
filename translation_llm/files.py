from pathlib import Path

from typing import Iterable, List, Union

PathInput = Union[str, Path]

def read_lines(path: PathInput, unescape_newline: bool = False) -> List[str]:
    """Reads lines from a file.

    Lines can be unescapped, meaning \\n is transformed to \n.
    
    Args:
        path: The path to the file.
        unescape_newline: Whether to unescape newlines.
        
    Returns:
        The lines in the file."""
    with open(path) as f:
        lines = [l[:-1] for l in f.readlines()]
    if unescape_newline:
        lines = [l.replace("\\n", "\n") for l in lines]
    return lines

 
def write_lines(
    path: PathInput, lines: Iterable[str], escape_newline: bool = False,
) -> None:
    """Writes lines to a file.

    Lines can be escaped, meaning \n is transformed to \\n.

    Args:
        path: The path to the file.
        lines: The lines to write.
        escape_newline: Whether to escape newlines.
    """
    if escape_newline:
        lines = (l.replace("\n", "\\n") for l in lines)
    with open(path, "w") as f:
        f.writelines((f"{l}\n" for l in lines))