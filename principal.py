from importlib import reload
from decimal import Decimal
from random import sample, uniform, randint
from datetime import datetime
import numpy as np
import pandas as pd
import logging
import problem
import evaluate
from problem import Problem
from evaluate import SolutionEvaluator


def log():
    logFormatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt='%m/%d %I:%M:%S')
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    
    if not rootLogger.handlers:
        fileHandler = logging.FileHandler(datetime.now().strftime('BBA_%d-%m_%H:%M.log'))
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)
    return rootLogger

rootLogger  = log()    

def minimize_comparator(individual_solution, global_solution) -> bool:
    return individual_solution < global_solution

def maximize_comparator(individual_solution, global_solution) -> bool:
    return individual_solution > global_solution

def create_population_uniform_strategy(X, num_particles):
    
    _, n_cols = X.shape
    lower = int((num_particles / 3) * 2)
    
    particles = np.zeros(shape=(num_particles, n_cols + 1))
    for i in range(lower):
        features = sample(range(n_cols), int(round(n_cols*0.2)))
        
        for j in features:
            particles[i,j] = 1
            
    for i in range(lower, num_particles):
        qtd_high = sample(range(n_cols), randint(round(n_cols/2 + 1), n_cols))
        
        for j in qtd_high:
            particles[i,j] = 1
        
    return particles


def create_population_20_50_strategy(X, num_particles):
    
    _, n_cols = X.shape
    particles = np.zeros(shape=(num_particles, n_cols + 1))    
    lower_group = int((num_particles / 3 ) * 2)
    
    for i in range(lower_group):
        features = sample(range(n_cols), int(round(n_cols * 0.2)))
    for j in features:
            particles[i,j] = 1
    
    for i in range(lower_group, num_particles):
        features = sample(range(n_cols), randint(round(n_cols / 2 + 1), n_cols))
        for j in features:
             particles[i,j] = 1
                                                 
    return particles                                             

def pulse_frequency_rate(num_particles):

    particles_loudness = np.zeros(shape=(num_particles, 1))
    particles_rate = np.zeros(shape=(num_particles, 1))

    for i in np.arange(0, num_particles):
        particles_loudness[i] = np.random.choice([0,1], 1)[0]
        particles_rate[i] = np.random.choice([1,2], 1)[0]

    return particles_loudness, particles_rate    


class BBAException(Exception):
                                                 
    def __init__(self,message):
        super(BBAException, self).__init__()
        self.message = message
                                                 
    def __str__(self):
        return repr(self.message)                                        

class BBASelector(object):
      
    def __init__(self, estimator, theta = 1.0, gamma = 1.0, epsilon = 1, num_particles=30, 
                    max_iter=100, max_local_improvement=50, maximize_objective=True, 
                    initialization='uniform', cv = 3):
                                                 
        self.theta = theta
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.cv = cv
        self.evaluator_ = None
        self.estimator = estimator
        self.velocity_ = None
        self.solution_ = None
        self.initialize = 0
        self.initialize_1 = 0
        self.maxfit = 0
        self.maxindex = 0
        self.N = None
        self.max_local_improvement = max_local_improvement
        self.local_improvement = 0
        self.particles = None
        self.count = []
        self.N_ = 0
        self.iteration_ = 0
        self.pop_ = None
        self.count_global = 0
        self._final_cols = None
        self._final_index = None
        self._setup_initialization(initialization)
        self._setup_solution_comparator(maximize_objective)
        self.selected_features_ = None
           
    def _setup_initialization(self, initialization):
           
        init_method = {
                'uniform': create_population_uniform_strategy,
                '20_50': create_population_20_50_strategy
                    }
                                                     
        self._initialization = initialization
        if initialization not in init_method:
            raise BBAException(f'Invalid method {initialization!r}')
        self.init_method_ = init_method[initialization]
        #self.init_search_ = init_search[type_search]
           
    def _setup_solution_comparator(self, maximize_objective):
        
        self.maximize_objective = maximize_objective
        if self.maximize_objective:
            self.is_solution_better = maximize_comparator
        else:
            self.is_solution_better = minimize_comparator
               
    def model_baseline(self, prob):

        n_cols = prob.data.shape[1]
        particle = np.zeros(shape=(1, n_cols + 1))
        particle[:] = 1
        evaluator_ = SolutionEvaluator(prob, 1)
        score = evaluator_.evaluate(particle)
        
        return score[0]

    def fit(self, X, unused_y, **kargs):
        
        if not isinstance(X, pd.DataFrame):
            raise BBAException('The "X" parameter must be a data frame')
            
        prob = Problem(X, unused_y, self.estimator,
                        self.cv, **kargs)
        
        self.N_ = prob.n_cols
        self._initialize(X)

        self.evaluator_ = SolutionEvaluator(prob, self.num_particles)

        score_all = self.model_baseline(prob)
        rootLogger.info((
                    f'Score with all features - {score_all[-1]}'))

        
        while not self._is_stop_criteria_accepted():
            self.init_search()

            count_sel_feat = self.count_features(self.best_global_[0])
            
            best_glob = self.best_global_[0]
            self.selected_features_  = np.ma.masked_where(best_glob[:-1]>0.6, best_glob[:-1])
            self.selected_features_, = np.where(self.selected_features_.mask == True)
            colunas = list(prob.data.iloc[:, self.selected_features_].columns)
            rootLogger.info((
                    f'Iteration: {self.iteration_}/{self.max_iter} \n , '
                    f'Best global metric: {self.best_global_[:, -1]} \n , '
                    f'Index features_selected: {self.selected_features_} \n , '
                    f'Number of selected features: {count_sel_feat} \n , '
                    f'Columns selected: {colunas}'))
            
            
            
        best_glob = self.best_global_[0]
        self.selected_features_ = np.ma.masked_where(best_glob[:-1]>0.6, best_glob[:-1])
        self.selected_features_, = np.where(self.selected_features_.mask == True)
        colunas = list(prob.data.iloc[:, self.selected_features_].columns)
        rootLogger.info((f'Final Index features selected: {self.selected_features_} /n, '
                        f'Final Columns selected: {colunas} \n'))
        
        self._final_cols = colunas     
        self._final_index = self.selected_features_
        
    def _initialize(self, X):
        
        self.iteration_ = 0
        self.pop_ = self.init_method_(X, self.num_particles)
        self.particles_loudness, self.particles_rate = pulse_frequency_rate(self.num_particles)
        self.particles_rate_ = np.zeros(shape=(self.num_particles, 1))
        self.velocity_ = np.zeros(shape=(self.num_particles, self.N_))
        self.best_individual_ =  np.zeros(shape=(self.num_particles, 1))
        self.best_global_ = np.zeros(shape=(1, self.N_ + 1))
        

    def _is_stop_criteria_accepted(self):
        
        no_global_improv = self.local_improvement >= self.max_local_improvement
        max_iter_reached = self.iteration_ >= self.max_iter
        return max_iter_reached or no_global_improv
    
    def init_search(self):
        
        self.pop_ = self.evaluator_.evaluate(self.pop_)
        self.evaluate_score(self.pop_)
        self.calculate_best_global()  
        self.bat_position()
        self.update_velocity()
        self.iteration_ += 1
            
    def evaluate_score(self, pop):

        if self.initialize == 0:
            for i in np.arange(0, len(pop)):
                self.best_individual_[i] = pop[i,-1]
                        
            self.initialize = 1
            
        for _i in np.arange(0, self.num_particles):
            rand = np.random.choice([0,1], 1)[0]
            if (rand < self.particles_loudness[_i]) & (pop[_i, -1] > self.best_individual_[_i]):
                self.best_individual_[_i] = pop[_i, -1]
                self.particles_loudness[_i] = self.theta * self.particles_loudness[_i]
                self.particles_rate_[_i] = self.particles_rate[_i] * (1 - np.exp(self.gamma * self.iteration_))

        self.maxfit, self.maxindex = np.max(self.best_individual_), np.argmax(self.best_individual_)
        
            
    def calculate_best_global(self):
        
        if self.initialize_1 == 0:
            for i in np.arange(0, self.N_ + 1):
                self.best_global_[0,i] = self.pop_[0, i]
                
            self.initialize_1 = 1

            if self.is_solution_better(self.maxfit,
                                    self.best_global_[0,-1]):

                for j in np.arange(0, self.N_ + 1):
                    self.best_global_[0,j] = self.pop_[self.maxindex,j]
                
    def bat_position(self):

        for _i in np.arange(0, self.num_particles):
            
            rand = np.random.choice([0,1], 1)[0]

            if rand > self.particles_rate_[_i]:
                for j in range(0, self.N_):
                    self.pop_[_i, j] =  self.pop_[_i, j] + self.epsilon * np.mean(self.particles_loudness)
                    sigma = uniform(0,1)
                    if sigma < 1 * ( 1 / (1 + np.exp(-self.pop_[_i, j]))):
                        self.pop_[_i, j] = 1
                    else:
                        self.pop_[_i, j] = 0

    def update_velocity(self):

        for _i in np.arange(0, self.num_particles):
            betha = np.random.choice([0,1], 1)[0]  
            rand = np.random.choice([0,1], 1)[0]
            if (rand < self.particles_loudness[_i]) & (self.best_individual_[_i] < self.best_global_[0, -1]):
                for j in range(0, self.N_):
                    fi = 0 + ( 0 + 1) * betha
                    self.velocity_[_i, j] = self.velocity_[_i, j] +  (self.best_global_[0, j] - self.pop_[_i, j]) * fi
                    self.pop_[_i, j] = self.pop_[_i, j] + self.velocity_[_i, j]
                    sigma = uniform(0,1)
                    if sigma < ( 1 / (1 + np.exp(-self.pop_[_i, j]))):
                        self.pop_[_i, j] = 1
                    else:
                        self.pop_[_i, j] = 0


    def count_features(self, particle_proportions, threshold=1):
         
        count = 0
        for i in range(0, self.N_):
            if particle_proportions[i] == threshold:
                count = count + 1
        return count 

    @property   
    def final_cols(self):           
        return self._final_cols 

    @property
    def final_index(self):
        return self._final_index    
              
              

            
                 
                 
                 
                
                
                
                
                
                
                
                
                     


            
                 
                 
                 
                 
                 
                 
                 
                 
                 
         
         
         
         
          
          
          
          
          
          
          
          
          
          
          
          
          
          
        
          
          
          
          
          
           
           
               
               
               
               
               
               
               
               
               
               
           
           
           
           
           
           
           
           
           


           
           
           
           
           
                                                 
                    
                    
                    
                                                 
                                                 