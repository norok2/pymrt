import pyparsing as pp
import doctest


# ======================================================================
def parse_name(text):
    """
    Parse XProtocol (a.k.a. EVP: EValuation Protocol)

    Args:
        text (str): Filename to be parsed

    Returns:
        parsed (dict): Result of the parsing.

    Examples:
        >>> text = 'S001__me+mp2rage_bw=280_t1_a=4'
        >>> parse_name(text)
    """
    g_sep = pp.Literal('_').suppress()
    kv_sep = pp.Literal('=').suppress()
    number = pp.Regex(r'[+-]?\d+(\.\d*)?').setName('number')
    special = '+-*/@'
    name = pp.Word(pp.alphanums + special).setName('name')
    s_id = pp.CaselessLiteral('s') + number
    # tag = pp.Dict(pp.Group(name)).setName('tag')
    tag = name

    key_value = pp.Dict(pp.Group(tag + kv_sep + number)).setName('key_value')
    exp = pp.Optional(s_id) + pp.OneOrMore(key_value | tag | g_sep)

    try:
        parsed = exp.parseString(text)
        result = parsed.asList(), parsed.asDict()
    except pp.ParseException:
        result = None
    return result


# ======================================================================
if __name__ == '__main__':
    print(__doc__)
    doctest.testmod()
