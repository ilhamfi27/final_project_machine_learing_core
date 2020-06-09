import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from skfeature.function.statistical_based import f_score
import tkinter
from tkinter import filedialog
from prepared.record_to_excel import record_it
from datetime import datetime

# current date and time
now = datetime.now()
right_now = now.strftime('%Y-%m-%d %H:%M:%S')

root = tkinter.Tk()
root.withdraw()

def main():
    import_file_path = filedialog.askopenfilename()
    sc = MinMaxScaler(feature_range=(0, 10))

    # 1. get data
    df = pd.read_excel(import_file_path)
    city_id = np.asarray(df['city_id'])
    raw_X = np.asarray(df.loc[:, 'sum_price_car':'std_buyer_land_rent']) # features
    raw_y = np.asarray(df['BPS_poverty_rate'])  # label

    # 2. pre-processing convert none to 0
    clean_X = np.nan_to_num(raw_X)
    clean_y = np.nan_to_num(raw_y)

    # 3. normalization
    sc.fit(raw_X)
    X = np.array(sc.transform(clean_X))
    y = np.array(clean_y)

    # 4. feature selection
    ranked_index = f_score.f_score(X, y, mode="index")
    ranked_index = ",".join(str(s) for s in ranked_index)
    record_it("f_score_record.xlsx", data=[[right_now, ranked_index]], columns=['time', 'ranked feature'])


if __name__ == "__main__":
    main()
