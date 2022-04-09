import pandas as pd


# ' download the data from the link without header and add the desired header to the data, as many
# '   date doesn't have header attached to it
# '
# '
# ' @param link: the link from the data without header will be downloaded
# ' @pparam header: the desired header to be added
# '
# ' @return return: dataframe contating the data and header
# '
# ' @export
# '
# ' @examples
# ' pre_process("https://raw.githubusercontent.com/jossiej00/iris_data/main/iris_raw.csv",
# '              ["Sepal.Length", "Sepal.Width", "Petal.Width", "Petal.Length", "Species"])

def pre_process(link, header):

    if isinstance(link, str) & isinstance(header, list):
        data = pd.read_csv(link, header=None)
        data.columns = header
        return data
    else:
        return ("Input 'link' should be a string type web link and input 'header' should be a list!")

