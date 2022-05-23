import re
def after_match(pattern, string):
    match = re.match(pattern, string)
    return string[match.span()[1]:]