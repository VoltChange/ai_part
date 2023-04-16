# This is a sample Python script.
from sentiment import SentimentAnalysis

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    score = SentimentAnalysis.analyse(u'这个实在是太好用了，我非常的喜欢，下次一定还会购买的！')
    print(score)