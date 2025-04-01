class Tensor:
    def __init__(self, dimension, data):
        self.dimension = dimension
        self.data = data
    
    def __repr__(self):
        return str(self.data)


class Matrix(Tensor):
    def __init__(self, dimension: tuple, data: list):
        super().__init__(dimension, data)
        self.rows = dimension[0]
        self.cols = dimension[1]
    
    def conv_rc2i(self, r: int, c: int) -> int:
        return r * self.cols + c
    
    def conv_i2rc(self, i: int) -> tuple[int, int]:
        return (i // self.cols, i % self.cols)
    
    def __str__(self):
        max_len = max(len(str(num)) for row in self.data for num in row)
        lines = ["["]
        for row in self.data:
            row_str = "  ".join(f"{num:>{max_len}}" for num in row)
            lines.append(row_str)
        lines.append("]")
        return "\n".join(lines)
    
    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            r, c = key
            if isinstance(r, int) and isinstance(c, int):
                return self.data[r][c]
        
        if isinstance(key, (int, slice)):
            new_data = [self.data[key]] if isinstance(key, int) else self.data[key]
            new_dim = (len(new_data), self.cols)
            return Matrix(new_dim, new_data)
        
        if isinstance(key, (list, tuple)):
            rows = []
            for k in key:
                if isinstance(k, int):
                    rows.append(self.data[k])
                elif isinstance(k, slice):
                    rows.extend(self.data[k])
            new_dim = (len(rows), self.cols)
            return Matrix(new_dim, rows)
        
        raise TypeError(f"Invalid key type: {type(key)}")
