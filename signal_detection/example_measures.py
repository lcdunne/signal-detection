if __name__=='__main__':

    import pandas as pd
    import measures

    df = pd.read_csv('../data/data.csv')

    x = measures.compute(data=df, y_true='y_true', y_resp='y_resp', group=['idcol', 'grouper'])

    print(x); print()

    print("d` [0.75, 0.21]", measures.d_prime(0.75, 0.21))
    print("C [0.75, 0.21]", measures.c_bias(0.75, 0.21))
    print("A` [0.75, 0.21]", measures.A_prime(0.75, 0.21))
    print("β [0.75, 0.21]", measures.beta(0.75, 0.21))
    print("β`` [0.75, 0.21]", measures.beta_doubleprime(0.75, 0.21))
    print("β`` (Donaldson) [0.75, 0.21]", measures.beta_doubleprime(0.75, 0.21, donaldson=True))
