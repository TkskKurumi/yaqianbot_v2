import re
def after_match(pattern, string):
    """
        Returns string after pattern.
        Example: after_match("/start", "/start with") -> " with"
    """
    match = re.match(pattern, string)
    return string[match.span()[1]:]