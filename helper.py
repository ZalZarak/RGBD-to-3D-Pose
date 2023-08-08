from cfonts import render as render_text
import time


def print_countdown(seconds):
    def print_in_big_font(text):
        rendered_text = render_text(text, gradient=['red', 'yellow'], align='center', size=(80, 40))
        print(rendered_text, end='\r')

    if seconds >= 1:
        for i in range(seconds, 0, -1):
            print_in_big_font(f"{i:02d}")
            time.sleep(1)

    print_in_big_font("GO!")
    print()  # Add a new line after the countdown ends