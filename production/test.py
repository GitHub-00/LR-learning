
if __name__ == '__main__':
    fp = open('../data/train_file')
    count = 0
    for line in fp:
        item = line.strip().split(',')
        print(len(item))
        count += 1
        if count >= 10:
            break