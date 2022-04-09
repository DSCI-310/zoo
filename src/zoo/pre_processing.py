import pandas as pd


# ' download the data from the link without header and add the desired header to the data, as many
# ' date doesn't have header attached to it

# ' @param link: the link from the data without header will be downloaded
# ' @param header: the desired header to be added
# ' @return: dataframe contating the data and header
# ' @examples
# ' pre_process(link, header)

def pre_process(link, header):

    data = pd.read_csv(link, header=None)
    data.columns = header

    return data
