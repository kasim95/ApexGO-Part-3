from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.agent.predict import DeepLearningAgent
from dlgo.networks.alphago import alphago_model

from keras.callbacks import ModelCheckpoint
import h5py

ROWS, COLS = 19, 19
NUM_CLASSES = ROWS * COLS
NUM_GAMES = 1000


def main():
    # sl data
    encoder = AlphaGoEncoder()
    processor = GoDataProcessor(encoder=encoder.name())

    # Paraller Processor
    generator = processor.load_go_data('train', NUM_GAMES, use_generator=True)
    test_generator = processor.load_go_data('test', NUM_GAMES, use_generator=True)


    # Data Processor
    # todo: does not have use_generator capability
    # generator = processor.load_go_data('train', NUM_GAMES)
    # test_generator = processor.load_go_data('test', NUM_GAMES)

    # sl model
    input_shape = (encoder.num_planes, ROWS, COLS)
    alphago_sl_policy = alphago_model(input_shape=input_shape, is_policy_net=True)
    alphago_sl_policy.compile(optimizer='sgd',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])

    # sl train
    epochs = 200
    batch_size = 128
    alphago_sl_policy.fit_generator(
        generator=generator.generate(batch_size, NUM_CLASSES),
        epochs=epochs,
        steps_per_epoch=generator.get_num_samples() / batch_size,
        validation_data=test_generator.generate(batch_size, NUM_CLASSES),
        validation_steps=test_generator.get_num_samples()/batch_size,
        callbacks=[ModelCheckpoint('alphago_sl_policy{epoch}.h5')]
    )
    alphago_sl_agent = DeepLearningAgent(alphago_sl_policy, encoder)

    # save model
    with h5py.File('alphago_sl_policy.h5', 'w') as sl_agent_out:
        alphago_sl_agent.serialize(sl_agent_out)

    # evaluate
    alphago_sl_policy.evaluate_generator(
        generator=test_generator.generate(batch_size, NUM_CLASSES),
        steps=test_generator.get_num_samples() / batch_size
    )


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
