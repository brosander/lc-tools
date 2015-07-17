import argparse
import csv
import json
import logging
import math
import MultiNEAT as NEAT
import os
import random
import shutil
import sys
import time

def outputFilename(outputDir, prefix, timestamp, extension):
  return '/'.join([outputDir, '.'.join([prefix, str(timestamp), extension])])

def jsonDump(value, outputDir, prefix, timestamp):
  with open(outputFilename(outputDir, prefix, timestamp, 'json'), 'w') as outFile:
      json.dump(value, outFile, indent=4, sort_keys=True)

def percentToFraction(percent):
  assert percent.endswith('%')
  splitPct = percent.split('.')
  decimal = splitPct[1][:-1]
  rawNum = splitPct[0] + decimal
  result = int(rawNum) / (100.0 * (math.pow(10, len(decimal))) )
  assert result > 0 and result < 0.4
  return result

def buildInputs(inputDir, logger):
  result = []
  valid_status = set(['Default', 'Fully Paid', 'Charged Off'])
  seen_status = set([])
  for potentialFile in os.listdir(inputDir):
    if potentialFile.endswith('.csv'):
      validRows = 0
      invalidRows = 0
      with open('/'.join([inputDir, potentialFile]), 'rb') as csvFile:
        csvFile.readline()
        csvReader = csv.DictReader(csvFile)
        for row in csvReader:
          int_rate_raw = row['int_rate']
          loan_status = row['loan_status']
          inq_last_6mths = row['inq_last_6mths']
          seen_status.add(loan_status)
          if int_rate_raw and inq_last_6mths and loan_status in valid_status:
            int_rate = percentToFraction(int_rate_raw)
            inq_last_6mths = int(inq_last_6mths)
            row['int_rate'] = int_rate
            row['inq_last_6mths'] = inq_last_6mths
            result.append((row, [int_rate, inq_last_6mths]))
            validRows += 1
          else:
            invalidRows += 1
      logger.info(potentialFile + ' had ' + str(validRows) + ' valid rows, ' + str(invalidRows) + ' invalid rows.')
  logger.info('Status codes encountered: ' + str(seen_status))
  result = [result[i] for i in sorted(random.sample(xrange(len(result)),1000))]
  winners = 0
  for inst in result:
    if inst[0]['loan_status'] == 'Fully Paid':
      winners += 1
  logger.info('Winners: ' + str(winners) + ' out of ' + str(len(result)))
  return result

def evaluate(genome, inputs):
  # this creates a neural network (phenotype) from the genome
  net = NEAT.NeuralNetwork()
  genome.BuildPhenotype(net)

  fitness = 0
  picks = ([], [])
  for inst in inputs:
    net.Input(inst[1])
    net.Activate()
    output = net.Output() 
    if output[0] >= 0.5:
      if inst[0]['loan_status'] == 'Fully Paid':
        fitness += inst[0]['int_rate']
        picks[0].append(inst)
      else:
        fitness -= 1
        picks[1].append(inst)
  return (fitness, picks[0], picks[1])

def runNeat(inputs, outputDir, logger, timestamp):
  max_fitness = -50000
  max_fitness_genome = None
  max_winners = None
  max_losers = None
  params = NEAT.Parameters()  
  genome = NEAT.Genome(0, len(inputs[0][1]), 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID, NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params)     
  pop = NEAT.Population(genome, params, True, 1.0)

  for generation in range(100): # run for 100 generations
    # retrieve a list of all genomes in the population
    genome_list = NEAT.GetGenomeList(pop)

    # apply the evaluation function to all genomes
    for genome in genome_list:
        fitness, winners, losers = evaluate(genome, inputs)
        genome.SetFitness(fitness)
        if fitness > max_fitness:
          logger.debug('Found new champion with fitness ' + str(fitness) + ' that picked ' + str(len(winners)) + ' winners and ' + str(len(losers)) + ' losers.')
          max_fitness = fitness
          max_fitness_genome = genome
          max_winners = winners
          max_losers = losers

    # at this point we may output some information regarding the progress of evolution, best fitness, etc.
    # it's also the place to put any code that tracks the progress and saves the best genome or the entire
    # population. We skip all of this in the tutorial. 

    # advance to the next generation
    pop.Epoch()
    logger.info('Done with generation: ' + str(generation))
  max_fitness_genome.Save('/'.join([outputDir, '.'.join(['maxFitnessGenome', str(timestamp), 'ge'])]))
  copyScriptSource = os.path.abspath(__file__)
  copyScriptDest = outputFilename(outputDir, 'lcNeat', timestamp, 'py')
  shutil.copyfile(copyScriptSource, copyScriptDest)
  jsonDump(max_winners, outputDir, 'winners', timestamp)
  jsonDump(max_losers, outputDir, 'losers', timestamp)
  jsonDump(inputs, outputDir, 'sample', timestamp)
  
def resolveDir(directory):
  return os.path.abspath(os.path.expanduser(directory))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='''
  This script is intended to run on the historical data from lending club
  ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("-i", "--inputDirectory", default=None, help="The input directory with the historical csvs")
  parser.add_argument("-o", "--outputDirectory", default=None, help="The output directory")
  args = parser.parse_args()

  if not args.inputDirectory:
    raise Exception('Must specify input directory')
  if not args.outputDirectory:
    raise Exception('Must specify output directory')

  timestamp = time.time()
  outputDir = resolveDir(args.outputDirectory)
  logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
  rootLogger = logging.getLogger()
 
  fileHandler = logging.FileHandler(outputFilename(outputDir, 'output', timestamp, 'log'))
  fileHandler.setFormatter(logFormatter)
  rootLogger.addHandler(fileHandler)
 
  consoleHandler = logging.StreamHandler()
  consoleHandler.setFormatter(logFormatter)
  rootLogger.addHandler(consoleHandler) 
  rootLogger.setLevel(logging.DEBUG)
  runNeat(buildInputs(resolveDir(args.inputDirectory), rootLogger), outputDir, rootLogger, timestamp)
