import csv
import json
import math
import MultiNEAT as NEAT
import os
import random
import shutil
import sys
import time

def percentToFraction(percent):
  assert percent.endswith('%')
  splitPct = percent.split('.')
  decimal = splitPct[1][:-1]
  rawNum = splitPct[0] + decimal
  result = int(rawNum) / (100.0 * (math.pow(10, len(decimal))) )
  assert result > 0 and result < 0.4
  return result

def buildInputs(inputDir):
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
          seen_status.add(loan_status)
          if int_rate_raw and loan_status in valid_status:
            int_rate = percentToFraction(int_rate_raw)
            row['int_rate'] = int_rate
            result.append((row, [int_rate]))
            validRows += 1
          else:
            invalidRows += 1
      print(potentialFile + ' had ' + str(validRows) + ' valid rows, ' + str(invalidRows) + ' invalid rows.')
  print('Status codes encountered: ' + str(seen_status))
  return [result[i] for i in sorted(random.sample(xrange(len(result)),1000))]

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

def runNeat(inputs, outputDir):
  timestamp = time.time()
  copyScriptSource = os.path.abspath(__file__)
  copyScriptDest = '/'.join([outputDir, '.'.join(['lcNeat', str(timestamp), 'py'])])
  print('copying ' + str(copyScriptSource) + ' to ' + str(copyScriptDest))
  shutil.copyfile(copyScriptSource, copyScriptDest)
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
          print('Found new champion with fitness ' + str(fitness) + ' that picked ' + str(len(winners)) + ' winners and ' + str(len(losers)) + ' losers.')
          max_fitness = fitness
          max_fitness_genome = genome
          max_winners = winners
          max_losers = losers

    # at this point we may output some information regarding the progress of evolution, best fitness, etc.
    # it's also the place to put any code that tracks the progress and saves the best genome or the entire
    # population. We skip all of this in the tutorial. 

    # advance to the next generation
    pop.Epoch()
    print('Done with generation: ' + str(generation))
  max_fitness_genome.Save('/'.join([outputDir, '.'.join(['maxFitnessGenome', str(timestamp), 'ge'])]))
  with open('/'.join([outputDir, '.'.join(['winners', str(timestamp), 'json'])]), 'w') as winnersFile:
    json.dump(max_winners, winnersFile, indent=4, sort_keys=True)
  with open('/'.join([outputDir, '.'.join(['losers', str(timestamp), 'json'])]), 'w') as losersFile:
    json.dump(max_losers, losersFile, indent=4, sort_keys=True)
  
if __name__ == '__main__':
  runNeat(buildInputs('/home/bryan/Downloads/historical'), '/home/bryan/Downloads/historical/neat')
