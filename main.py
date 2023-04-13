import yaml
from module.parser import getParser
from module.log import printLog
from module.data_preprocess import dataPreprocess
from module.model_utils import runModel

if __name__ == "__main__":
    args = getParser()
    with open('config/' + args.config_file + '.yaml', 'r') as stream:
        config = yaml.load(stream, Loader = yaml.Loader)

    printLog("Start ProTrack.")

    if config['module']['data_preprocess']:
        printLog("Start preprocessing data.")
        dataPreprocess(config)

    if config['module']['run_model']:
        printLog("Start running model.")
        runModel(config)

    printLog("End ProTrack.")