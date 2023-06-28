from deap import base, creator, tools, algorithms
import random

class GeneticAlgorithm:
    def __init__(self, metric_calculator, n_population=50, n_iterations=100):
        self.metric_calculator = metric_calculator
        self.n_population = n_population
        self.n_iterations = n_iterations

    def evaluate(self, individual):
        # Transform binary individual to a list of boolean values
        mask = [bool(gene) for gene in individual]

        # Calculate the difference between distributions using the provided metric calculator
        return self.metric_calculator.calculate(mask),

    def run(self):
        # Create types
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Initialize toolbox
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(self.metric_calculator.data.columns))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Register genetic operators
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Initialize population
        population = toolbox.population(n=self.n_population)

        # Run the algorithm
        for gen in range(self.n_iterations):
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            population = toolbox.select(offspring, k=len(population))

        # Return the top 10 individuals from the last population
        return tools.selBest(population, k=10)
      
