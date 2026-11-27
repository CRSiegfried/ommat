# Records docs/assets/demo.gif, the terminal recording at the top of the README.
# From the repository root, after `uv sync`: vhs docs/assets/demo/demo.tape
# It needs vhs, ttyd and ffmpeg on the PATH.

Output docs/assets/demo.gif

Set Shell bash
Set FontSize 20
Set Width 1100
Set Height 640
Set Padding 26
Set TypingSpeed 28ms
Set Theme "Catppuccin Mocha"

Hide
Type "source .venv/bin/activate && cd docs/assets/demo && rm -rf records && export PS1='$ ' && clear"
Enter
Sleep 1500ms
Show

Sleep 400ms
Type `python assistant.py "Find my recent emails from Neo or Smith"`
Sleep 500ms
Enter
Sleep 8s

Hide
Type "clear"
Enter
Sleep 300ms
Show
Type `python assistant.py "Can you deal with that thing from yesterday"`
Sleep 500ms
Enter
Sleep 6500ms

Hide
Type "clear"
Enter
Sleep 300ms
Show
Type "python assistant.py --replay"
Sleep 500ms
Enter
Sleep 4500ms

Hide
Type "rm -rf records __pycache__"
Enter
Sleep 300ms
