def wrap_letters(text):
    html = ""
    for char in text:
        html += (
            f'<span class="letter-wrapper">'
            f'<span class="letter top">{char}</span>'
            f'<span class="letter bottom">{char}</span>'
            f'</span>'
        )
    return html