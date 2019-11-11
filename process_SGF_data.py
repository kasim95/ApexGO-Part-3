from dlgo.data.data_processor import GoDataProcessor


def main():
    processor = GoDataProcessor()
    features, labels = processor.load_go_data('train', 100)
    features_test, labels_test = processor.load_go_data('test', 20)


if __name__ == '__main__':
    main()
