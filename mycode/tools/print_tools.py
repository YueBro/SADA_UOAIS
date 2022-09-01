from typing import List

class _Styler:
    def __init__(self):
        self._style_dict = {
            "default": "0",
            "bold": "1",
            "italic": "3",
            "under": "4",
            "underline": "4",
            "blink": "5",
            "highlight": "7",
            "cross": "9",
        }

        self._col_dict = {
            "default": "0",
            "black": "0",
            "grey": "90",
            "red": "91",
            "green": "92",
            "yello": "93",
            "blue": "94",
            "pink": "95",
            "cyan": "96",
        }

        self._default_keys = ["reset", "default"]
        self._default = "\033[0m"
    
    def __getattr__(self, s):
        if s in self._default_keys:
            return self._default
        
        s = s.split("_")
        return self.get(*s)
    
    def __getitem__(self, s):
        if isinstance(s, str):
            s = (s,)
        if not isinstance(s, tuple):
            raise SyntaxError("Incorrect input format!")
        
        return self.get(*s)
    
    def __call__(self, *args):
        return self.get(*args)

    def get(self, *s):
        if len(s)==1 and s[0] in self._default_keys:
            return self._default
        
        col_count_down = 1

        elements = []
        for e in s:
            if e in self._style_dict.keys():
                elements.append(self._style_dict[e])
            elif e in self._col_dict.keys() and col_count_down > 0:
                elements.append(self._col_dict[e])
                col_count_down -= 1
            else:
                raise SyntaxError("Incorrect input format!")
        return "\033[" + ";".join(elements) + "m"
    
    def stylize(self, string: str, *styles):
        return self.get(*styles) + string + self.get('reset')
    

styler = _Styler()


def table_creater(table_contents, sep=None, hline_above_indices=[0, 1, -1]):
    # Get parameters
    elements = []
    lengths = []
    for i in range(len(table_contents)):
        sub_elements = []
        sub_lengths = []
        for j in range(len(table_contents[i])):
            try:
                element = table_contents[i][j].__str__()
            except Exception:
                element = table_contents[i][j].__repr__()
            sub_elements.append(element)
            sub_lengths.append(len(element)+2)
        elements.append(sub_elements)
        lengths.append(sub_lengths)
    
    n_row = len(table_contents)
    n_col = max(map(len, elements))
    if sep is None:
        sep = ["|"] * (n_col+1)

    # Parameter adjustments
    col_lengths = [0] * n_col
    for i in range(n_row):
        for j in range(len(lengths[i])):
            if lengths[i][j] > col_lengths[j]:
                col_lengths[j] = lengths[i][j]
    hline_above_indices = [(n_row if (i==-1) else i) for i in hline_above_indices]
    hline_above_indices.sort()

    # Error check
    assert len(sep)==(n_col+1), "len(sep) must equal to maximum length of \"table_contents\"."
    assert not(min(hline_above_indices)<-1), f"\"hline_above_indices\" contains unknown index {min(hline_above_indices)}."
    
    # Print
    s = ""
    for i in range(n_row+1):
        if hline_above_indices != [] and i == hline_above_indices[0]:
            s += "-" * (sum(col_lengths)+sum(map(len, sep))) + "\n"
            hline_above_indices.pop(0)
        
        if i < n_row:
            s += sep[0]
            for j in range(n_col):
                try:
                    e = table_contents[i][j].__str__()
                except Exception as exception:
                    if type(exception) == AttributeError:
                        e = table_contents[i][j].__repr__()
                    elif type(exception) == IndexError:
                        e = ""
                    else:
                        raise exception
                s += ("{:^"+str(col_lengths[j])+"}").format(e) + sep[j+1]
            s += "\n"
    
    # Return
    return s
