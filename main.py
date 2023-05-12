# This is a sample Python script.
from sentiment import SentimentAnalysis

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    comments = u'这个很棒，我很喜欢！'
    score = SentimentAnalysis.analyse(comments)
    print(comments)
    print(score)