import pandas as pd

def data_wrangling():
    # Updated the original file to the recent version from World Bank (last update: 2019-12-20)
    df = pd.read_csv('data/API_SP.RUR.TOTL.ZS_DS2_en_csv_v2_718759.csv', skiprows = 4)

    # Filter for 1990 and 2018, top 10 economies
    df = df[['Country Name','1990', '2018']]
    countrylist = ['United States', 'China', 'Japan', 'Germany', 'United Kingdom', 'India', 'France', 'Brazil', 'Italy', 'Canada']
    df = df[df['Country Name'].isin(countrylist)]

    # melt year columns  and convert year to date time
    df_melt = df.melt(id_vars='Country Name', value_vars = ['1990', '2018'])
    df_melt.columns = ['country','year', 'variable']
    df_melt['year'] = df_melt['year'].astype('datetime64[ns]').dt.year

    # add column names
    df_melt.columns = ['country', 'year', 'percentrural']

    # prepare data into x, y lists for plotting
    df_melt.sort_values('percentrural', ascending=False, inplace=True)

    data = []
    for country in countrylist:
        x_val = df_melt[df_melt['country'] == country].year.tolist()
        y_val =  df_melt[df_melt['country'] == country].percentrural.tolist()
        data.append((country, x_val, y_val))
        print(country, x_val, y_val)
        
    return data