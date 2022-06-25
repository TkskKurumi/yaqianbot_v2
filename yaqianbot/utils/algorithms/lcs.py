from dataclasses import dataclass, asdict
from ..image.colors import Color
def ndarray(dims, fill=0):
    if(len(dims) == 1):
        n = dims[0]
        return [fill for i in range(n)]
    else:
        return [ndarray(dims[1:], fill=fill) for i in range(dims[0])]
def element_similarity(a, b):
    if(a.lower()==b.lower()):
        return 1
    else:
        return 0
def colored(st, fore=None, back=None):
    ls = []
    if(fore is not None):
        ls.append(Color.from_any(fore).as_terminal_fg())
    if(back is not None):
        ls.append(Color.from_any(back).as_terminal_bg())
    ls.append(st)
    if(fore or back):
        ls.append(Color().as_terminal_rst())
    return "".join(ls)
@dataclass
class _lcs:
    A: str
    B: str
    common: str
    common_ratio_a: float
    common_ratio_b: float
    common_ratio: float
    common_len: int
    a_matched: list
    b_matched: list

    def calc(A, B):
        global _debug
        n = len(A)
        m = len(B)
        dp = ndarray((n, m))
        a_matched = ndarray((n,), False)
        b_matched = ndarray((m,), False)
        dp_from = ndarray((n, m), (-1, -1))
        for i in range(n):
            for j in range(m):
                '''if(A[i] in 'Aa' and B[j] in 'Aa' and A[i]!=B[j]):
                    print(A[i],B[j],A[i].lower() == B[j].lower())'''
                mx = 0
                _dp_from = (-1, -1)

                # match A[i], B[j]
                score = dp[i-1][j-1] if (i and j) else 0
                score1 = element_similarity(A[i], B[j])

                if(score1):

                    if(score+score1 >= mx):
                        mx = score+score1
                        _dp_from = (i-1, j-1)
                if(i):
                    if(dp[i-1][j] >= mx):
                        mx = dp[i-1][j]
                        _dp_from = (i-1, j)
                if(j):
                    if(dp[i][j-1] >= mx):
                        mx = dp[i][j-1]
                        _dp_from = (i, j-1)
                dp[i][j] = mx
                dp_from[i][j] = _dp_from
        u, v = n-1, m-1
        common = []
        while(u >= 0 and v >= 0):

            u1, v1 = dp_from[u][v]
            ''' if(_debug):
                print(u, v, 'from', u1, v1) '''
            if(u1 == u-1 and v1 == v-1):
                if(element_similarity(A[u], B[v]) > 0.5):
                    common.append(A[u])
                    '''if(_debug):
                        print("matching", A[u], B[v])'''
                    a_matched[u] = True
                    b_matched[v] = True
            u, v = u1, v1

        common = common[::-1]
        '''#self.A = A
        self.B = B
        self.common = common  # list
        common_ratio_a = dp[n-1][m-1]/len(A)
        common_ratio_b = dp[n-1][m-1]/len(B)
        self.common_ratio = self.common_ratio_a*self.common_ratio_b
        self.common_len = dp[n-1][m-1]'''
        common_len = dp[n-1][m-1]
        common_ratio_a = common_len/len(A)
        common_ratio_b = common_len/len(B)
        return _lcs(A, B, common, common_ratio_a, common_ratio_b, common_ratio_a*common_ratio_b, common_len, a_matched, b_matched)

    def color_common(self, foreA="RED", foreB="GREEN"):
        retA = []
        for idx, i in enumerate(self.A):
            if(self.a_matched[idx]):
                retA.append(str(colored(i, fore=foreA)))
            else:
                retA.append(i)
        retA = "".join(retA)

        retB = []
        for idx, i in enumerate(self.B):
            if(self.b_matched[idx]):
                retB.append(str(colored(i, fore=foreB)))
            else:
                retB.append(i)
        retB = "".join(retB)
        return retA, retB

    def asdict(self, preserve_AB=False):
        D = asdict(self)
        if(not preserve_AB):
            D.pop("A")
            D.pop("B")
        D["a_matched"] = "".join(['1' if i else '0' for i in self.a_matched])
        D["b_matched"] = "".join(['1' if i else '0' for i in self.b_matched])
        return D

    def fromdict_A_B(D, A, B):
        D['a_matched'] = [bool(int(i)) for i in D["a_matched"]]
        D['b_matched'] = [bool(int(i)) for i in D["b_matched"]]
        return _lcs(A=A, B=B, **D)

def lcs(A, B):
    
    return _lcs.calc(A, B)
if(__name__=="__main__"):
    A = "被戳的反应"
    B = "被戳戳嗯喵咩"
    l = lcs(A, B)
    print(*l.color_common())
    print(l.common_len)