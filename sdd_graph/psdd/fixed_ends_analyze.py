#!/usr/bin/env python

import math
import time
import glob
import sys
import os.path
import random
import pickle
import heapq

from collections import defaultdict
from pypsdd import *

# for printing numbers with commas
import locale
locale.setlocale(locale.LC_ALL, "en_US.UTF8")

class PathManager(object):
    def __init__(self,rows,cols,edge2index,edge_index2tuple,fl2models_fn=None,copy=None,fl2prediction_fn=None):
        self.rows = rows
        self.cols = cols
        self.num_nodes = rows*cols
        self.num_edges = (rows-1) * cols + (cols-1) * rows
        self.edge2index = edge2index
        self.edge_index2tuple = edge_index2tuple
        self.paths = []
        self.copy = copy
        if fl2models_fn != None:
            self.fl2models = pickle.load(open(fl2models_fn,'rb'))
        if fl2prediction_fn != None:
            self.fl2prediction = pickle.load(open(fl2prediction_fn,'rb'))


    def all_all_predictions(self):
        """Find and save all the all-at-once predictions for a given grid.
        only gets prediction for (min(i,j),max(i,j)) since prediction for (i,j) == prediction for (j,i)
        """
        fl2all_prediction = {}
        count = 0
        for fl in self.fl2models:
            key = (min(fl[0],fl[1]),max(fl[0],fl[1]))
            if key not in fl2all_prediction:
                print count
                count += 1
                all_prediction = self.best_all_at_once(fl[0],fl[1])
                fl2all_prediction[key] = all_prediction
            if count > 100:
                break
        """
        for i in range(1,self.num_nodes+1):
            for j in range(i+1,self.num_nodes+1):
                print (i,j)
                all_prediction = self.best_all_at_once(i,j)
                fl2all_prediction[(i,j)] = all_prediction
                reverse_prediction = self.best_all_at_once(j,i) 
                for k in range(len(all_prediction)):
                    if all_prediction[k] != reverse_prediction[k]:
                        print "reverse not same! %s" % str((i,j))
                """
                #fl2all_prediction[(j,i)] = fl2all_prediction[(i,j)]

        #with open('pickles/first_last2all_prediction_taken-%d-%d.pickle' % (self.rows,self.cols),'wb') as output:
        #with open('pickles/first_last2all_prediction-%d-%d.pickle' % (self.rows,self.cols),'wb') as output:
        with open('pickles/first_last2all_prediction_100_filter_more-%d-%d.pickle' % (self.rows,self.cols),'wb') as output:
            pickle.dump(fl2all_prediction,output)

        

    def model_matches_fl(self,model,fl):
        """Determine if the model is consistent with the actual start and end point of the represented trip.
        Return True if the model terminates at start and end.
        Return False otherwise.
        """
        first = fl[0]
        second = fl[1]
        first_incidents = self.incident_edges(first)
        second_incidents = self.incident_edges(second)
        if exactly_one(first_incidents,model) and exactly_one(second_incidents,model):
            return True
        #print fl
        #self.draw_grid(model)
        return False

    def testing_fl2models(self):
        """Create a dict mapping fl to the models to the trip_ids for that model for the testing dataset"""
        trip_id2in = pickle.load(open('better_pickles/t2testing.pickle','rb'))
        trip_id2fl = pickle.load(open('../pickles/trip_id2first_last-%d-%d.pickle' % (self.rows,self.cols),'rb'))
        fl2models = {}
        trip_id2model = pickle.load(open('better_pickles/trip_id2model.pickle','rb'))
        inserted = 0
        for t in trip_id2in:
            fl = trip_id2fl[t]
            if fl not in fl2models:
                fl2models[fl] = defaultdict(list)
            model = trip_id2model[t]
            fl2models[fl][model].append(t)
            inserted += 1
        with open('better_pickles/testing_fl2models.pickle','wb') as output:
            pickle.dump(fl2models,output)



    def create_fl2models(self,data_fn,bad_fn):
        """Create a dictionary that maps a (first,last) tuple to the path models taken to get from first to last.
        Map those models to the trip_ids of paths that take the model.

        This function also determines the bad paths as defined by not having a path consistent with ones first, last pair.
            We do not save this right now, but use it in calculating the fl2models to have the true models
        """
        trip_id2in = pickle.load(open('better_pickles/trip_id2good.pickle','rb'))
        trip_id2fl = pickle.load(open('../pickles/trip_id2first_last-%d-%d.pickle' % (self.rows,self.cols),'rb'))
        fl2models = {}
        trip_id2model = pickle.load(open('better_pickles/trip_id2model.pickle','rb'))
        bad_paths = {}
        inserted = 0
        for t in trip_id2in:
            fl = trip_id2fl[t]
            if fl not in fl2models:
                fl2models[fl] = defaultdict(list)
            model = trip_id2model[t]
            if inserted < 25:
                print t
                print fl
                self.draw_grid(model)
                print ""
            fl2models[fl][model].append(t)
            inserted += 1
                
        """
        with open(bad_fn,'r') as infile:
            bad_indices = map(int,infile.readlines())
            for index in bad_indices:
                bad_paths[index+1] = True
        with open(data_fn,'r') as infile:
            lines = infile.readlines()
            full_ints = map(lambda x: map(int,x[:-1].split(',')),lines)
            full_tuple = map(tuple,full_ints)
        #to deal with the 1-indexing of trip ids
        full_tuple.insert(0,0)
        fl2models = {}
        inserted = 0
        for trip_id in range(1,len(full_tuple)):
            #if trip_id > 25:
            #    return
            if (trip_id) not in bad_paths:
                #print "inserting trip: %d" % trip_id
                trip_fl = trip_id2fl[trip_id]
                #print "trip first last: %s" % str(trip_fl)
                model = full_tuple[trip_id]
                if not self.model_matches_fl(model,trip_fl) or trip_fl[0] == trip_fl[1]:
                    #if trip_id < 200:
                    #    print trip_id
                    bad_paths[trip_id] = True
                    continue
                if trip_fl not in fl2models:
                    fl2models[trip_fl] = defaultdict(list)
                #self.draw_grid(model)
                fl2models[trip_fl][model].append(trip_id)
                inserted += 1
        print "num inserted: %d" % inserted
        self.fl2models = fl2models
        """
        with open('better_pickles/first_last2models-%d-%d.pickle' % (self.rows,self.cols),'wb') as output:
            pickle.dump(fl2models,output)
        #with open('pickles/trip_id2bad-%d-%d.pickle' % (self.rows,self.cols),'wb') as output:
            #pickle.dump(bad_paths,output)

    def analyze_predictions_new(self):
        """Find similarity measures for the predicted paths
        Weight measures by the frequency of a given path (# of trips for that path, not first last pair)
        """
        radii = [3,6]
        num_dists = len(radii) + 1
        #fl2prediction = pickle.load(open('better_pickles/fl2prediction.pickle','rb'))
        dist2num_trips = defaultdict(float)
        dist2haus = defaultdict(float)
        dist2ampsd = defaultdict(float)
        dist2dsn = defaultdict(float)
        dist2correct_guess = defaultdict(float)
        total_trips = 0.0
        tot_haus = 0.0
        tot_ampsd = 0.0
        tot_dsn = 0.0
        correctly_guessed = 0.0
        fl_pairs_examined = 0
        for first_last in self.fl2prediction:
            prediction = self.fl2prediction[first_last]
            distance = self.node_dist(first_last[0],first_last[1])
            dist = len(radii)
            for i in range(len(radii)):
                if distance <= radii[i]:
                    dist = i
                    break
            for fl in (first_last,(first_last[1],first_last[0])):
                models = None
                if fl in self.fl2models:
                    models = self.fl2models[fl]
                else:
                    continue
                fl_pairs_examined += 1
                for model in models:
                    model_count = len(models[model])
                    total_trips += model_count
                    dist2num_trips[dist] += model_count
                    haus,ampsd,dsn = self.path_diff_measures(model,prediction)
                    print "%s: haus %.2f, ampsd %.2f, dsn %.2f" % (str(fl),haus,ampsd,dsn) 
                    dist2haus[dist] += model_count*haus
                    dist2ampsd[dist] += model_count*ampsd
                    dist2dsn[dist] += model_count*dsn
                    tot_haus += model_count*haus
                    tot_ampsd += model_count*ampsd
                    tot_dsn += model_count*dsn
                    if dsn == 0:
                        correctly_guessed += model_count
                        dist2correct_guess[dist] += model_count

        for i in range(num_dists):
            num_trips = dist2num_trips[i]
            if num_trips == 0:
                print "No paths for group %d" % i
            dist2haus[i] = dist2haus[i]/num_trips
            dist2ampsd[i] = dist2ampsd[i]/num_trips
            dist2dsn[i] = dist2dsn[i]/num_trips
            dist2correct_guess[i] = dist2correct_guess[i]/num_trips
            print ""
            if i == 0:
                print "0 <= Radius <= %d" % radii[0]
            elif i < len(radii):
                print "%d < Radius <= %d" % (radii[i-1],radii[i])
            else:
                print "%d < Radius" % (radii[-1])
            print "Correctly guessed %.2f percent of trips" % (100.0*dist2correct_guess[i])
            print "%d total trips" % num_trips
            print "average hausdorff %.2f, average ampsd %.2f, average dsn %.2f" % (dist2haus[i],dist2ampsd[i],dist2dsn[i])

        avg_haus = tot_haus/total_trips
        avg_ampsd = tot_ampsd/total_trips
        avg_dsn = tot_dsn/total_trips
        correct_pct = correctly_guessed/total_trips
        print "\nOverall"
        print "Examined %d first last pairs" % fl_pairs_examined
        print "Average Hausdorff Distance: %.3f" % avg_haus
        print "Average Average Minimum Point Segment Distance Distance: %.3f" % avg_ampsd
        print "Average dsn: %.3f" % avg_dsn
        print "Correctly guessed %.2f percent of trips" % (100.0*correct_pct)

    def analyze_predictions(self):
        """Find similarity measures for the predicted paths
        Weight measures by the frequency of a given path (# of trips for that path, not first last pair)
        """
        radii = [3,6]
        fl2prediction = pickle.load(open('pickles/first_last2all_prediction_taken-10-10.pickle','rb'))
        #fl2prediction = pickle.load(open('pickles/first_last2all_prediction_100_filter_more-10-10.pickle','rb'))
        #fl2prediction = pickle.load(open('pickles/first_last2all_prediction_some-10-10.pickle','rb'))
        total_trips = 0.0
        tot_haus = 0.0
        tot_ampsd = 0.0
        tot_dsn = 0.0
        correctly_guessed = 0.0
        fl_pairs_examined = 0
        for first_last in self.fl2prediction:
            prediction = self.fl2prediction[first_last]
            for fl in (first_last,(first_last[1],first_last[0])):
                models = None
                if fl in self.fl2models:
                    models = self.fl2models[fl]
                else:
                    continue
                fl_pairs_examined += 1
                for model in models:
                    model_count = len(models[model])
                    total_trips += model_count
                    haus,ampsd,dsn = self.path_diff_measures(model,prediction)
                    print "%s: haus %.2f, ampsd %.2f, dsn %.2f" % (str(fl),haus,ampsd,dsn) 
                    tot_haus += model_count*haus
                    tot_ampsd += model_count*ampsd
                    tot_dsn += model_count*dsn
                    if dsn == 0:
                        correctly_guessed += model_count
        avg_haus = tot_haus/total_trips
        avg_ampsd = tot_ampsd/total_trips
        avg_dsn = tot_dsn/total_trips
        correct_pct = correctly_guessed/total_trips
        print "Examined %d first last pairs" % fl_pairs_examined
        print "Average Hausdorff Distance: %.3f" % avg_haus
        print "Average Average Minimum Point Segment Distance Distance: %.3f" % avg_ampsd
        print "Average dsn: %.3f" % avg_dsn
        print "Correct guess percentage: %.3f" % correct_pct


    def print_some_instances(self):
        count = 0
        for fl in self.fl2models:
            count += 1
            if count % 400 == 0:
                self.understand_similarity(fl)

    def find_better_prediction(self):
        good_count = 0.0
        total_count = 0.0
        for fl in self.fl2prediction:
            if fl not in self.fl2models:
                continue
            if self.prediction_better(fl):
                good_count += 1
            total_count += 1
        print "Prediction was better: %.2f percent of the time" % (good_count/total_count)


    def prediction_better(self,fl):
        """Find the S,E pairs where the prediction has better similarity than the most frequent model"""
        model2ts = self.fl2models[fl]
        if fl not in self.fl2prediction:
            return
        prediction = self.fl2prediction[fl]
        best_model,best_score = most_frequent_model(model2ts)
        for model in model2ts:
            m_count = len(model2ts[model])
        haus,ampsd,dsn = self.evaluate_prediction_vs_models(best_model,model2ts)
        haus_p,ampsd_p,dsn_p = self.evaluate_prediction_vs_models(prediction,model2ts)
        if haus_p < haus or ampsd_p < ampsd or dsn_p < dsn:
            #print fl
            #self.understand_similarity(fl)
            return True

    def number_guess_top_model(self):
        total = 0.0
        guessed_top = 0.0
        for fl in self.fl2prediction:
            if fl not in self.fl2models:
                continue
            total += 1
            model2ts = self.fl2models[fl]
            best_model,best_score = most_frequent_model(model2ts)
            prediction = self.fl2prediction[fl]
            same = True
            for i in range(len(prediction)):
                if prediction[i] != best_model[i]:
                    same = False
            if same == True:
                guessed_top += 1
        print guessed_top
        best_pct = guessed_top/total
        print "Pct top %.3f" % best_pct


    def understand_similarity(self,fl):
        """A method used to understand how our similarity measurements work."""
        model2ts = self.fl2models[fl]
        if fl not in self.fl2prediction:
            return
        prediction = self.fl2prediction[fl]
        best_model,best_score = most_frequent_model(model2ts)
        for model in model2ts:
            m_count = len(model2ts[model])
            print m_count
            self.draw_grid(model)
            print ""
        print "BEST MODEL"
        print best_score
        self.draw_grid(best_model)
        haus,ampsd,dsn = self.evaluate_prediction_vs_models(best_model,model2ts)
        print "haus %.2f, ampsd %.2f, dsn %.2f" % (haus,ampsd,dsn) 
        print "PREDICTION"
        self.draw_grid(prediction)
        haus_p,ampsd_p,dsn_p = self.evaluate_prediction_vs_models(prediction,model2ts)
        print "haus %.2f, ampsd %.2f, dsn %.2f" % (haus_p,ampsd_p,dsn_p) 

        
    def evaluate_prediction_vs_models(self,prediction,model2ts):
        """Find the weighted average of the similarity measures between the prediction and the models."""
        haus=ampsd=dsn=0.0
        tot_trips = 0.0
        for model in model2ts:
            count = len(model2ts[model])
            tot_trips += count
            haus_i,ampsd_i,dsn_i = self.path_diff_measures(model,prediction)
            haus += count*haus_i
            ampsd += count*ampsd_i
            dsn += count*dsn_i
        haus = haus/tot_trips
        ampsd = ampsd/tot_trips
        dsn = dsn/tot_trips
        return haus,ampsd,dsn

    def visualize_similarities(self,fl):
        models = self.fl2models[fl]
        model_array = []
        num_models = len(models)
        probs = [0.0 for i in range(num_models)]
        total_trips = 0.0
        model_i = 0
        for model in models:
            count = len(models[model])
            probs[model_i] += count
            total_trips += count
            model_array.append(model)
            model_i += 1
        if len(model_array) == 1:
            return

        for i in range(len(model_array)):
            for j in range(i+1,len(model_array)):
                model_1 = model_array[i]
                model_2 = model_array[j]
                haus,ampsd,dsn = self.path_diff_measures(model_1,model_2)
                self.draw_grid(model_1)
                self.draw_grid(model_2)
                print "%s: haus %.2f, ampsd %.2f, dsn %.2f" % (str((i,j)),haus,ampsd,dsn) 



    def compare_observed_models_new(self):
        """Compare the difference between different paths taken for the same start,end pair.
        Weight these differences by the proportion of paths that took a given model.
        Radii are distances for splitting paths based on segment distance from first to last.
            first class is 0 <= dist <= radii[0]
            last class is radii[-1] < dist
        """
        radii = [3,6]
        num_dists = len(radii)+1
        fl2dist_class = {}
        num_iters = 0
        fl2num_trips = {}
        #first element is hausdorff distance, second is sum hausdorff, third is dsn
        #these are measurements over all combinations of two different paths for a given fl
        #   pick two paths at random.  If they are the same, pick two paths at random again.
        fl2similarity_measures_mult = {}
        dist2tot_trips_mult = defaultdict(float)
        #these are the measurements for all fl pairs and all models. Do not only examine differing paths
        fl2similarity_measures = {}
        dist2tot_trips = defaultdict(float)

        dist2num_models = defaultdict(float)
        tot_models = 0.0

        for fl in self.fl2models:
            dist = self.node_dist(fl[0],fl[1])
            dist_class = len(radii)
            for i in range(len(radii)):
                if dist <= radii[i]:
                    dist_class = i
                    break
            fl2dist_class[fl] = dist_class
            models = self.fl2models[fl]
            num_models = len(models)
            probs = [0.0 for i in range(len(models))]
            model_array = []
            total_trips = 0.0
            model_i = 0
            for model in models:
                count = len(models[model])
                probs[model_i] += count
                total_trips += count
                model_array.append(model)
                #print "Trips with model %d: %d" % (model_i,count)
                model_i += 1
            dist2num_models[dist_class] += num_models*total_trips
            tot_models += num_models*total_trips
            dist2tot_trips[dist_class] += total_trips
            fl2num_trips[fl] = total_trips
            fl2similarity_measures[fl] = [0.0,0.0,0.0]
            if len(model_array) == 1:
                for i in range(3):
                    fl2similarity_measures[fl][i] = 0.0
                continue
            dist2tot_trips_mult[dist_class] += total_trips
            probs = map(lambda x: x/total_trips,probs)
            diag_sum = sum(map(lambda x: x*x,probs))
            denom = 1.0-diag_sum
            weights = [[0.0 for i in range(num_models)] for i in range(num_models)]
            for i in range(num_models):
                for j in range(i+1,num_models):
                    weights[i][j] = (2*probs[i]*probs[j])/denom

            """Calculate weighted similarity measures for different path measurements"""
            fl2similarity_measures_mult[fl] = [0.0,0.0,0.0]
            for i in range(len(model_array)):
                for j in range(i+1,len(model_array)):
                    weight = weights[i][j]
                    haus,ampsd,dsn = self.path_diff_measures(model_array[i],model_array[j])
                    #print "%s: haus %.2f, ampsd %.2f, dsn %.2f" % (str((i,j)),haus,ampsd,dsn) 
                    fl2similarity_measures_mult[fl][0] += weight*haus
                    fl2similarity_measures_mult[fl][1] += weight*ampsd
                    fl2similarity_measures_mult[fl][2] += weight*dsn
            measures = fl2similarity_measures_mult[fl]
            #print "Diff path overall: haus %.2f, ampsd %.2f, dsn %.2f" % (measures[0],measures[1],measures[2])
            """Reconfigure weights to correspond to all possible combinations"""
            weights_with_diag = [[0.0 for i in range(num_models)] for i in range(num_models)]
            for i in range(num_models):
                for j in range(i,num_models):
                    if i == j:
                        weights_with_diag[i][j] = probs[i]*probs[i]
                    else:
                        weights_with_diag[i][j] = weights[i][j]*denom
            """Calculate weighted similarity measures for any two paths, can be the same"""
            weight_sum = 0.0
            for i in range(num_models):
                weight_sum += sum(weights_with_diag[i])
            #print "weight sum: %f" % weight_sum
            for i in range(len(model_array)):
                for j in range(i,len(model_array)):
                    weight = weights_with_diag[i][j]
                    haus,ampsd,dsn = self.path_diff_measures(model_array[i],model_array[j])
                    #print "%s: haus %.2f, ampsd %.2f, dsn %.2f" % (str((i,j)),haus,ampsd,dsn) 
                    fl2similarity_measures[fl][0] += weight*haus
                    fl2similarity_measures[fl][1] += weight*ampsd
                    fl2similarity_measures[fl][2] += weight*dsn
            measures = fl2similarity_measures[fl]

            #print "overall: haus %.2f, ampsd %.2f, dsn %.2f\n" % (measures[0],measures[1],measures[2])
            num_iters += 1
        dist2haus = defaultdict(float)
        dist2ampsd = defaultdict(float)
        dist2dsn = defaultdict(float)
        dist2haus_mult = defaultdict(float)
        dist2ampsd_mult = defaultdict(float)
        dist2dsn_mult = defaultdict(float)
        tot_haus = 0.0
        tot_ampsd = 0.0
        tot_dsn = 0.0
        tot_haus_mult = 0.0
        tot_ampsd_mult = 0.0
        tot_dsn_mult = 0.0
        tot_mult_trips = 0.0
        tot_trips = 0.0
        for fl in fl2num_trips:
            num_trips = fl2num_trips[fl]
            tot_trips += num_trips
            dist_class = fl2dist_class[fl]
            if len(self.fl2models[fl]) > 1:
                mult_meas = fl2similarity_measures_mult[fl]
                weighted_haus = num_trips*mult_meas[0]
                weighted_ampsd = num_trips*mult_meas[1]
                weighted_dsn = num_trips*mult_meas[2] 
                dist2haus_mult[dist_class] += weighted_haus 
                dist2ampsd_mult[dist_class] += weighted_ampsd 
                dist2dsn_mult[dist_class] += weighted_dsn 
                tot_haus_mult += weighted_haus
                tot_ampsd_mult += weighted_ampsd
                tot_dsn_mult += weighted_dsn
                tot_mult_trips += num_trips
            meas = fl2similarity_measures[fl]
            weighted_haus = num_trips*meas[0]
            weighted_ampsd = num_trips*meas[1]
            weighted_dsn = num_trips*meas[2] 
            dist2haus[dist_class] += weighted_haus
            dist2ampsd[dist_class] += weighted_ampsd
            dist2dsn[dist_class] += weighted_dsn
            tot_haus += weighted_haus
            tot_ampsd += weighted_ampsd
            tot_dsn += weighted_dsn
        for i in range(num_dists):
            num_trips_mult = dist2tot_trips_mult[i]
            num_trips = dist2tot_trips[i]
            dist2num_models[i] = dist2num_models[i]/num_trips
            dist2haus_mult[i] = dist2haus_mult[i]/num_trips_mult
            dist2ampsd_mult[i] = dist2ampsd_mult[i]/num_trips_mult
            dist2dsn_mult[i] = dist2dsn_mult[i]/num_trips_mult
            dist2haus[i] = dist2haus[i]/num_trips
            dist2ampsd[i] = dist2ampsd[i]/num_trips
            dist2dsn[i] = dist2dsn[i]/num_trips
            print ""
            if i == 0:
                print "0 <= Radius <= %d" % radii[0]
            elif i < len(radii):
                print "%d < Radius <= %d" % (radii[i-1],radii[i])
            else:
                print "%d < Radius" % (radii[-1]+1)
            print "average number of models per fl pair: %.2f" % dist2num_models[i]
            print "%d trips for pairs with multiple paths" % num_trips_mult
            print "%d total trips" % num_trips
            print "Diff paths average hausdorff %.2f, average ampsd %.2f, average dsn %.2f" % (dist2haus_mult[i],dist2ampsd_mult[i],dist2dsn_mult[i])
            print "average hausdorff %.2f, average ampsd %.2f, average dsn %.2f" % (dist2haus[i],dist2ampsd[i],dist2dsn[i])

        tot_models = tot_models/tot_trips
        tot_haus_mult = tot_haus_mult/tot_mult_trips
        tot_ampsd_mult = tot_ampsd_mult/tot_mult_trips
        tot_dsn_mult = tot_dsn_mult/tot_mult_trips
        tot_haus = tot_haus/tot_trips
        tot_ampsd = tot_ampsd/tot_trips
        tot_dsn = tot_dsn/tot_trips
        print ""
        print "Overall"
        print "average number of models per fl pair: %.2f" % tot_models
        print "Diff paths average hausdorff %.2f, average ampsd %.2f, average dsn %.2f" % (tot_haus_mult,tot_ampsd_mult,tot_dsn_mult)
        print "average hausdorff %.2f, average ampsd %.2f, average dsn %.2f" % (tot_haus,tot_ampsd,tot_dsn)
        return


    def compare_observed_models(self):
        """Compare the difference between different paths taken for the same start,end pair.
        Weight these differences by the proportion of paths that took a given model.
        """
        num_iters = 0
        tot_ovr_trips_mult_paths = 0.0
        fl2num_trips = {}
        #first element is hausdorff distance, second is sum hausdorff, third is dsn
        fl2similarity_measures = {}
        for fl in self.fl2models:
            models = self.fl2models[fl]
            num_models = len(models)
            probs = [0.0 for i in range(len(models))]
            model_array = []
            total_trips = 0.0
            model_i = 0
            for model in models:
                count = len(models[model])
                probs[model_i] += count
                total_trips += count
                model_array.append(model)
                #print "Trips with model %d: %d" % (model_i,count)
                model_i += 1
            if len(model_array) == 1:
                continue
            tot_ovr_trips_mult_paths += total_trips
            fl2num_trips[fl] = total_trips
            probs = map(lambda x: x/total_trips,probs)
            diag_sum = sum(map(lambda x: x*x,probs))
            denom = 1.0-diag_sum
            weights = [[0.0 for i in range(num_models)] for i in range(num_models)]
            for i in range(num_models):
                for j in range(i+1,num_models):
                    weights[i][j] = (2*probs[i]*probs[j])/denom
           # """
            fl2similarity_measures[fl] = [0.0,0.0,0.0]
            for i in range(len(model_array)):
                for j in range(i+1,len(model_array)):
                    weight = weights[i][j]
                    haus,sum_haus,dsn = self.path_diff_measures(model_array[i],model_array[j])
                    #print "%s: haus %.2f, sum_haus %.2f, dsn %.2f" % (str((i,j)),haus,sum_haus,dsn) 
                    fl2similarity_measures[fl][0] += weight*haus
                    fl2similarity_measures[fl][1] += weight*sum_haus
                    fl2similarity_measures[fl][2] += weight*dsn
            measures = fl2similarity_measures[fl]
            #"""
            """
            for i in range(len(model_array)):
                print "path %d" % i
                self.draw_grid(model_array[i])
            weights_with_diag = [[0.0 for i in range(num_models)] for i in range(num_models)]
            for i in range(num_models):
                for j in range(i,num_models):
                    if i == j:
                        weights_with_diag[i][j] = probs[i]*probs[i]
                    else:
                        weights_with_diag[i][j] = weights[i][j]*denom
            fl2similarity_measures[fl] = [0.0,0.0,0.0]
            weight_sum = 0.0
            for i in range(num_models):
                #for j in range(num_models):
                #    sys.stdout.write("%.3f " % weights_with_diag[i][j])
                #print ""
                weight_sum += sum(weights_with_diag[i])
            #print "weight sum: %f" % weight_sum
            for i in range(len(model_array)):
                for j in range(i,len(model_array)):
                    weight = weights_with_diag[i][j]
                    haus,sum_haus,dsn = self.path_diff_measures(model_array[i],model_array[j])
                    #print "%s: haus %.2f, sum_haus %.2f, dsn %.2f" % (str((i,j)),haus,sum_haus,dsn) 
                    fl2similarity_measures[fl][0] += weight*haus
                    fl2similarity_measures[fl][1] += weight*sum_haus
                    fl2similarity_measures[fl][2] += weight*dsn
            measures = fl2similarity_measures[fl]
            """
            #print "overall: haus %.2f, sum_haus %.2f, dsn %.2f" % (measures[0],measures[1],measures[2])
            #print ""
            #if num_iters > 6:
            #    break
            num_iters += 1
        overall_haus = 0.0
        overall_sum_haus = 0.0
        overall_dsn = 0.0
        for fl in fl2num_trips:
            if len(self.fl2models[fl]) == 1:
                continue
            num_trips = fl2num_trips[fl]
            meas = fl2similarity_measures[fl]
            overall_haus += num_trips*meas[0]
            overall_sum_haus += num_trips*meas[1]
            overall_dsn += num_trips*meas[2]
        overall_haus = overall_haus/tot_ovr_trips_mult_paths
        overall_sum_haus = overall_sum_haus/tot_ovr_trips_mult_paths
        overall_dsn = overall_dsn/tot_ovr_trips_mult_paths
        print "\naverage hausdorff %.2f, average sum hausdorff %.2f, average dsn %.2f" % (overall_haus,overall_sum_haus,overall_dsn)
        return

    def analyze_paths_taken(self):
        """Find statistics on how many different paths are taken for a given frst,last pair."""
        count_and_fl_long = []
        radius = 6 
        #this will double count overlapping paths going from (i,j) and (j,i)
        total_paths = 0
        total_fl_pairs = 0
        total_long_pairs = 0
        weighted_total_long_paths = 0
        total_long_paths = 0
        total_long_trips = 0
        total_trips = 0
        weighted_total_paths = 0
        for fl in self.fl2models:
            models = self.fl2models[fl]
            total_fl_pairs += 1
            num_paths = len(self.fl2models[fl])
            total_paths += num_paths
            num_trips = 0
            for model in models:
                num_trips += len(models[model])
            total_trips += num_trips
            weighted_total_paths += num_trips*num_paths

            if self.node_dist(fl[0],fl[1]) > radius:
                total_long_trips += num_trips
                total_long_pairs += 1
                total_long_paths += num_paths
                weighted_total_long_paths += num_trips*num_paths
                heapq.heappush(count_and_fl_long,[(0-num_paths),fl])
        print "total paths: %d" % total_paths
        print "average paths per fl pair: %f" % (float(total_paths)/total_fl_pairs)
        print "weighted average number of paths per fl pair: %f" % (float(weighted_total_paths)/(total_trips))

        print "\nRadius %d" % radius

        print "total long pairs (min distance %d): %d" % (radius,total_long_pairs)
        print "average paths per long fl pair: %f" % (float(total_long_paths)/total_long_pairs)
        print "weighted average number of paths per long fl pair: %f" % (float(weighted_total_long_paths)/(total_long_trips))
        quarter_of_fl_long = total_long_pairs/4
        for i in range(4):
            count,fl = heapq.heappop(count_and_fl_long)
            count = 0-count
            print "%dth percentile has %d models for long paths (min radius %d)" % ((100-i*25),count,radius)
            print "first, last: %s" % str(fl)
            """
            for model in self.fl2models[fl]:
                self.draw_grid(model)
                print ""
            """
            if i == 3:
                break
            for j in range(quarter_of_fl_long):
                fl = heapq.heappop(count_and_fl_long)[1]
                """
                if j < 2:
                    print "first, last: %s" % str(fl)
                    for model in self.fl2models[fl]:
                        self.draw_grid(model)
                        print ""
                """

    def node_dist(self,node1,node2):
        return euclidean(self.node_to_tuple(node1),self.node_to_tuple(node2))

    def incident_edges(self,node):
        neighbors = self.neighbor_nodes(node)
        return map(lambda x: self.edge2index[min(x,node),max(x,node)],neighbors)

    def nearest_neighbor(self,point,coords2in):
        """Find the euclidean distance of the nearest neighbor to point in the model

        Attributes
            point: (x,y)
            coords2in: dict mapping (i,j) to True if it is included in the model
        """
        row,col = point
        best_dist = self.rows
        step = 0
        while step < best_dist:
            for row_i in range(row-step,row+step+1):
                if row_i < 0 or row_i >= self.rows:
                    continue
                for col_i in (col-step,col+step):
                    if col_i < 0 or col_i >= self.cols:
                        continue
                    if (row_i,col_i) in coords2in:
                        dist = euclidean(point,(row_i,col_i))
                        if dist < best_dist:
                            best_dist = dist
            for col_i in range(col-step+1,col+step):
                if col_i < 0 or col_i >= self.cols:
                    continue
                for row_i in (row-step,row+step):
                    if row_i < 0 or row_i >= self.rows:
                        continue
                    if (row_i,col_i) in coords2in:
                        dist = euclidean(point,(row_i,col_i))
                        if dist < best_dist:
                            best_dist = dist
            step += 1
        return best_dist

    def edge_array_to_coords(self,edge_array):
        """Determines all nodes traversed in an edge path.

        Attributes
            edge_array: an array of length num_edges that has traversed edges set to 1 (other edges 0 or -1)
        Returns
            node2in: dict mapping traversed nodes (in coordinate form) to True
        """
        coords2in = {}
        for i in range(self.num_edges):
            if edge_array[i] == 1:
                node1,node2 = self.edge_index2tuple[i]
                coords2in[self.node_to_tuple(node1)] = True
                coords2in[self.node_to_tuple(node2)] = True
        return coords2in

    def max_and_total_shortest(self,coords2in1,coords2in2):
        max_shortest = 0.0
        total_shortest = 0.0
        for point in coords2in1:
            nearest = self.nearest_neighbor(point,coords2in2)
            total_shortest += nearest
            if nearest > max_shortest:
                max_shortest = nearest
        return max_shortest,total_shortest


    def path_diff_measures(self,edge_path1,edge_path2):
        haus,ampsd = self.min_and_sum_hausdorff(edge_path1,edge_path2)
        dsn = edge_dsn(edge_path1,edge_path2)
        """
        print "Hausdorff: %f" % haus
        print "Sum Hausdorff: %f" % ampsd
        print "Dissimilarity: %f" % dsn
        """
        return haus,ampsd,dsn


    def min_and_sum_hausdorff(self,edge_path1,edge_path2):
        """Find the hausdorff and sum hausdorff distance for two edge arrays.
        
        Hausdorff distance is max nearest neighbor distance over every point in path 1

        Sum Hausdorff is (sum of nearest neighbors for points in path 1)/points in path 1 + (same for path 2) all divided by 2

        Attributes
            edge_path1: array that has traversed edges set to 1
            edge_path2: same
        Returns:
            Hausdorff and sum hausdorff distances
        """
        coords2in1 = self.edge_array_to_coords(edge_path1)
        coords2in2 = self.edge_array_to_coords(edge_path2)
        worst1,total1 = self.max_and_total_shortest(coords2in1,coords2in2)
        worst2,total2 = self.max_and_total_shortest(coords2in2,coords2in1)
        worst = max(worst1,worst2)
        normed_total = ((total1/len(coords2in1)) + (total2/len(coords2in2)))/2
        return worst,normed_total


    def save_paths(self,start,end,step_by_step,all_at_once=None):
        """Save path instantiations to files in the paths directory."""

        step_fn = "paths/step_%d_%d_%d_%d.pickle" % (self.rows,self.cols,start,end)
        with open(step_fn,'wb') as output:
            pickle.dump(step_by_step,output)

        if all_at_once != None:
            all_fn = "paths/all_%d_%d_%d_%d.pickle" % (self.rows,self.cols,start,end)

            with open(all_fn,'wb') as output:
                pickle.dump(all_at_once,output)


    def draw_grid(self,model):
        m = self.rows
        n = self.cols
        for i in xrange(n):
            sys.stdout.write("%d " % i)
        sys.stdout.write("\n")
        for i in xrange(m):
            for j in xrange(n):
                sys.stdout.write('.')
                if j < n-1:
                    edge = (i*m+j+1,i*m+j+2)
                    index = self.edge2index[edge]
                    sys.stdout.write('-' if model[index] == 1 else ' ')
            sys.stdout.write(' %d\n' % i)
            if i < m-1:
                for j in xrange(n):
                    edge = (i*m+j+1,i*m+m+j+1)
                    index = self.edge2index[edge]
                    sys.stdout.write('|' if model[index] == 1 else ' ')
                    sys.stdout.write(' ')
            sys.stdout.write('\n')

    def draw_edge_probs(self,model,edge_num2prob,start,end):
        m = self.rows
        n = self.cols
        for i in xrange(m):
            sys.stdout.write("  ")
            for j in xrange(n):
                if i*n + j + 1 == start:
                    sys.stdout.write('s')
                elif i*n + j + 1 == end:
                    sys.stdout.write('e')
                else:
                    sys.stdout.write('.')
                if j < n-1:
                    edge = (i*m+j+1,i*m+j+2)
                    index = self.edge2index[edge]
                    if model[index] == 1:
                        sys.stdout.write(' ----- ')
                    elif index in edge_num2prob.keys():
                        if edge_num2prob[index] > 0.999:
                            sys.stdout.write("   1   ")
                        else:
                            sys.stdout.write(' %s ' % ('%.4f' % edge_num2prob[index])[1:])
                    else:
                        sys.stdout.write('       ')
            sys.stdout.write('\n')
            if i < m-1:
                for j in xrange(n):
                    edge = (i*m+j+1,i*m+m+j+1)
                    index = self.edge2index[edge]
                    if model[index] == 1:
                        sys.stdout.write('  |     ')
                    elif index in edge_num2prob.keys():
                        if edge_num2prob[index] > 0.999:
                            sys.stdout.write("  1     ")
                        else:
                            sys.stdout.write('%s   ' % ('%.4f' % edge_num2prob[index])[1:])
                    else:
                        sys.stdout.write('        ')
            sys.stdout.write('\n')

    def most_likely_start(self,start,end):
        """Find the most likely start edge given the start node and end node.

        Return:
            edge to be set to true
            list of edges to be set to false
        """
        start_asgnmts = self.end_point(start)
        end_asgnmts = self.end_point(end)

        start_end_prob = self.prob_start_end(start,end)

        edge_num2prob = {}
        best_prob = 0.0
        best_i = 0
        for s_i in range(len(start_asgnmts)):
            total_prob = 0.0
            s_a = start_asgnmts[s_i]
            s_edge = s_a[0]
            for e_i in range(len(end_asgnmts)):
                e_a = end_asgnmts[e_i]
                p = Path(self)
                p.add_and_neg_edges([e_a[0]],e_a[1])
                p.add_and_neg_edges([s_a[0]],s_a[1])
                #print p.model_tuple()
                evidence = DataSet.evidence(p.model_tuple())
                path_prob = self.copy.probability(evidence)
                #p.ones_and_zeros()
                #print "path probability normalized: %.6f" % (path_prob/start_end_prob)
                total_prob += path_prob
            normalized_prob = total_prob/start_end_prob
            edge_num2prob[s_edge] = normalized_prob
            #print "total prob: %.6f" % total_prob
            #print "Probability of taking edge %d: %.6f" % (start_asgnmts[s_i][0],total_prob/start_end_prob)
            if total_prob > best_prob:
                best_prob = total_prob
                best_i = s_i


        self.draw_edge_probs([-1 for i in range(self.num_edges)],edge_num2prob,start,end)
        return start_asgnmts[best_i]

    def partial_prob(self,partial,end):
        """Find the probability of a path with the given partial path and an end at node end.

        Sum over all possible ends.

        Return the probability.
        """

        total_prob = 0.0
        end_asgnmts = self.end_point(end)
        for end_asgnmt in end_asgnmts:
            path = Path(self,partial[:])
            path.add_and_neg_edges([end_asgnmt[0]],end_asgnmt[1])
            evidence = DataSet.evidence(path.model_tuple())
            path_prob = self.copy.probability(evidence)
            total_prob += path_prob

        return total_prob



    def most_likely_next(self,node,prev_node,start,end,cur_path):
        """Find the most likely next edge to be taken conditioned on end point and current path.

        Attributes:
            node: current node
            prev_node: previous visited node
            end: end node
            cur_path: instantiations of -1,0,1 for the path (equivalent to a Path.model)

        Return the edge index to be set to positive.
        """
        neighbors = self.neighbor_nodes(node)
        end_asgnmts = self.end_point(end)

        possible_edges = []
        for neighbor in neighbors:
           if neighbor != prev_node:
               possible_edges.append(self.edge2index[(min(neighbor,node),max(neighbor,node))])
        
        partial_prob = self.partial_prob(cur_path[:],end)
        edge_num2prob = {}
        best_prob = 0.0
        best_i = 0
        for s_i in range(len(possible_edges)):
            total_prob = 0.0
            edge = possible_edges[s_i]
            for e_i in range(len(end_asgnmts)):
                e_a = end_asgnmts[e_i]
                p = Path(self,cur_path[:])
                if p.add_and_neg_edges([e_a[0]],e_a[1]) == -1:
                    continue
                if p.add_edge(edge) == -1:
                    continue
                
                evidence = DataSet.evidence(p.model_tuple())
                path_prob = self.copy.probability(evidence)
                total_prob += path_prob
            normalized_prob = total_prob/partial_prob
            edge_num2prob[edge] = normalized_prob
            #print "total prob: %.6f" % total_prob
            #print "Probability of taking edge %d: %.6f" % (start_asgnmts[s_i][0],total_prob/start_end_prob)
            if total_prob > best_prob:
                best_prob = total_prob
                best_i = s_i


        print ""
        print ""
        self.draw_edge_probs(cur_path,edge_num2prob,start,end)
        return possible_edges[best_i]



    def best_all_at_once(self,start,end):
        end_asgnmts = self.end_point(end)
        start_asgnmts = self.end_point(start)


        best_model = None
        best_prob = 0.0
        for s_a in start_asgnmts:
            for e_a in end_asgnmts:
                inst = [-1 for i in range(self.num_edges)]
                inst[s_a[0]] = 1
                inst[e_a[0]] = 1
                for neg_edge in s_a[1]:
                    inst[neg_edge] = 0
                for neg_edge in e_a[1]:
                    inst[neg_edge] = 0
                evidence = DataSet.evidence(inst)
                try:
                    val,model = self.copy.mpe(evidence)
                except:
                    print "start_edge: %d" % s_a[0]
                    print "end_edge: %d" % e_a[0]
                    continue
                mpe_model = [0 for i in range(self.num_edges)]
                for key in model:
                    if model[key] == 1:
                        mpe_model[key-1] = 1
                #print "\n"
                #self.draw_grid(mpe_model)
                if val > best_prob:
                    best_prob = val
                    best_model = mpe_model

        data = tuple(best_model)
        return best_model

        


    def best_step_by_step(self,start,end):
        """Find the most likely path to be taken between start and end"""
        path = Path(self)
        cur_node = start

        pos_edge,neg_edges = self.most_likely_start(start,end)

        if path.add_and_neg_edges([pos_edge],neg_edges) == -1:
            print "INVALID PATH"
            return -1

        prev_node = start
        incident_nodes = self.edge_index2tuple[pos_edge]
        if cur_node == incident_nodes[0]:
            cur_node = incident_nodes[1]
        else:
            cur_node = incident_nodes[0]

        while cur_node != end:
            next_edge = self.most_likely_next(cur_node,prev_node,start,end,path.model[:])
            prev_node = cur_node
            incident_nodes = self.edge_index2tuple[next_edge]
            if cur_node == incident_nodes[0]:
                cur_node = incident_nodes[1]
            else:
                cur_node = incident_nodes[0]
            if path.add_edge(next_edge) == -1:
                print "INVALID ADDING EDGE %d" % next_edge

        print "\n"
        for i in range(len(path.model)):
            if path.model[i] == -1:
                path.model[i] = 0
        self.draw_edge_probs(path.model[:],{},start,end)
        #print "STEP-BY-STEP PREDICTION"
        #self.draw_grid(path.model)

        return path.model[:]


                

    def tuple_to_node(self,point):
        """Given a node represented as a tuple return corresponding node index"""
        return self.cols*point[0] + point[1] + 1

    def node_to_tuple(self,node_num):
        """Given a node index, return corresponding node tuple"""
        row = (node_num-1) / self.cols
        col = (node_num-1) % self.cols
        return (row,col)

    def out_edges(self,node):
        """Find all the edes that pass out of a node.

        Return:
            up: edge index passing out of the top of the node (-1 if in top row)
            down: edge index passing out of bottom (-1 if bottom row)
            left: edge index passing out of left (-1 if left col)
            right: edge index passing out of right (-1 if right col)
        """
        up = -1
        down = -1
        left = -1
        right = -1
        if node > self.cols:
            up = self.edge2index[(node-self.cols,node)]
        if node <= self.cols*(self.rows-1):
            down = self.edge2index[(node,node+self.cols)]
        if node % self.cols != 1:
            left = self.edge2index[(node-1,node)]
        if node % self.cols != 0:
            right = self.edge2index[(node,node+1)]

        return up,down,left,right


    def draw_all_paths(self):
        for path in self.paths:
            self.draw_grid(path.model)
            print ""

    def neighbor_nodes(self,node):
        """Find all the nodes that neighbor a node.

        Return:
            A list of all the node indexes that neighbor the given node.
        """

        neighbors = []
        if node > self.cols:
            neighbors.append(node-self.cols)
        if node <= self.cols*(self.rows-1):
            neighbors.append(node+self.cols)
        if node % self.cols != 1:
            neighbors.append(node-1)
        if node % self.cols != 0:
            neighbors.append(node+1)

        return neighbors

    def add_path(self,model):
        self.paths.append(Path(self,model))

    def end_point(self,node):
        """Find all the variables that should be set to represent a path ending at node.

        There are at most 4 sets of variable assignments to do this.
            Each one has 1 edge set to true and the other (1-3) edges set to false.

        Return a list
            Elements of the top-level list are len 2 lists
                First element of second level list is the edge to be set to true
                Second element of second level is list of edges to be set to false
        """
        assignments = []

        up,down,left,right = self.out_edges(node)

        for pos_edge in up,down,left,right:
            if pos_edge != -1:
                asn = [pos_edge,[]]
                for neg_edge in up,down,left,right:
                    if neg_edge != pos_edge and neg_edge != -1:
                        asn[1].append(neg_edge)
                assignments.append(asn)

        return assignments

    def mid_point(self,node):
        """Find all the variable assignments representing a path passing through node
        
        These are all valid pairs of 2 edges connected to the node.

        Return
            A list of lists of length 2.
            Each sublist is 2 edges taken.

        DO A SANITY CHECK THAT SETTING EXTRA EDGES TO 0 DOES NOTHIG TO PROBABILITY
        """
        assignments = []
        up,down,left,right = self.out_edges(node)

        valid_edges = []
        for i in up,down,left,right:
            if i != -1:
                valid_edges.append(i)

        for i in range(len(valid_edges)):
            for j in range(i+1,len(valid_edges)):
                assignments.append([valid_edges[i],valid_edges[j]])

        return assignments

    def start_set(self,start,end):
        start_asgnmts = self.end_point(start)
        end_asgnmts = self.end_point(end)

        for i in range(len(start_asgnmts)):
            for j in range(len(end_asgnmts)):
                bad_path = False
                p = Path(self)
                s_a = start_asgnmts[i]
                e_a = end_asgnmts[j]
                p.add_edge(s_a[0])
                p.add_edge(e_a[0])
                for neg_edge in s_a[1]:
                    if p.negate_edge(neg_edge) == -1:
                        bad_path = True
                        break
                for neg_edge in e_a[1]:
                    if p.negate_edge(neg_edge) == -1:
                        bad_path = True
                        break
                if bad_path == False:
                    self.paths.append(p)


    def prob_start_end_mid(self,start,end,mid):
        """Probability that a path starts at start and ends at end and passes through mid.
        
        This value is NOT normalized by the probability that a path starts at star and ends at end.

        Return that probability as a float.
        """

        start_asgnmts = self.end_point(start)
        end_asgnmts = self.end_point(end)
        mid_asgnmts = self.mid_point(mid)

        total_prob = 0.0
        for start_i in range(len(start_asgnmts)):
            for end_i in range(len(end_asgnmts)):
                for mid_i in range(len(mid_asgnmts)):
                    start_a = start_asgnmts[start_i]
                    end_a = end_asgnmts[end_i]
                    mid_a = mid_asgnmts[mid_i]
                    if start_a[1].count(mid_a[0]) != 0 or end_a[1].count(mid_a[0]) != 0:
                        continue
                    if start_a[1].count(mid_a[1]) != 0 or end_a[1].count(mid_a[1]) != 0:
                        continue
                            
                    data = [-1 for i in range(self.num_edges)]
                    for one in mid_a:
                        data[one] = 1
                    data[start_a[0]] = 1
                    data[end_a[0]] = 1
                    for zero in start_a[1]:
                        data[zero] = 0
                    for zero in end_a[1]:
                        data[zero] = 0
                    ones = []
                    zeros = []
                    for i in range(len(data)):
                        if data[i] == 0:
                            zeros.append(i)
                        if data[i] == 1:
                            ones.append(i)
                    #print "start edge: %d, end edge: %d, mid edges: %d %d" % (start_a[0],end_a[0],mid_a[0],mid_a[1])
                    #print "ones: %s" % str(ones)
                    #print "zeros: %s" % str(zeros)
                    data = tuple(data)
                    evidence = DataSet.evidence(data)
                    probability = self.copy.probability(evidence)
                    #print "prob: %f" % probability
                    total_prob += probability
                    
        return total_prob 

    def normalized_prob_mid(self,start,end,mid):
        """Probability that a path starts at start, ends at end, and passes through mid, normalized.
        Normalizing factor is probability that the path starts at start and ends at end.
        Return normalized probability.
        """
        mid_prob = self.prob_start_end_mid(start,end,mid)
        start_end_prob = self.prob_start_end(start,end)
        #print "start end prob %d: %f" % (mid,start_end_prob)
        #print "mid prob %d: %f" % (mid,mid_prob)
        return mid_prob/start_end_prob

    def prob_start_end(self,start,end):
        """Probability that a path starts at start and ends at end.

        Reutrn that probability as a float.
        """

        start_asgnmts = self.end_point(start)
        end_asgnmts = self.end_point(end)

        total_prob = 0.0
        for start_i in range(len(start_asgnmts)):
            for end_i in range(len(end_asgnmts)):
                start_a = start_asgnmts[start_i]
                end_a = end_asgnmts[end_i]
                data = [-1 for i in range(self.num_edges)]
                data[start_a[0]] = 1
                data[end_a[0]] = 1
                for zero in start_a[1]:
                    data[zero] = 0
                for zero in end_a[1]:
                    data[zero] = 0
                data = tuple(data)
                evidence = DataSet.evidence(data)
                probability = self.copy.probability(evidence)
                total_prob += probability

        return total_prob

    def visualize_mid_probs(self,start,end):
        probs = []
        for i in range(self.rows):
            for j in range(self.cols):
                mid = i*self.cols + j + 1
                if mid == start:
                    sys.stdout.write("start   ")
                    continue
                if mid == end:
                    sys.stdout.write(" end    ")
                    continue
                prob_mid = self.normalized_prob_mid(start,end,mid)
                sys.stdout.write("%.3f   " % prob_mid)
            sys.stdout.write("\n\n")
        
class Path(object):
    """An object to hold and manipulate variable instantiations for a psdd.
    """
    def __init__(self,manager,model=None):
        self.manager = manager
        if model == None:
            self.model = [-1 for i in range(self.manager.num_edges)]
        else:
            self.model = model[:]

    def model_tuple(self):
        return tuple(self.model)

    def ones_and_zeros(self):
        ones = []
        zeros = []
        for i in range(self.manager.num_edges):
            if self.model[i] == 0:
                zeros.append(i)
            elif self.model[i] == 1:
                ones.append(i)

        print "zeros: %s" % str(zeros)
        print "ones: %s" % str(ones)

    def add_and_neg_edges(self,to_add,to_neg):
        """add all the edges in to_add and negate all the edges in to_neg.
        If any operation is invalid, return -1.
        Otherwise return 1.
        """

        for i in to_add:
            if self.add_edge(i) == -1:
                return -1
        for j in to_neg:
            if self.negate_edge(j) == -1:
                return -1
        return 1

    def negate_edge(self,edge_num):
        """Change the model to reflect negating an edge.

        Set model[edge_num] to 0 if legal to do so (edge hasn't been set to 1 yet).

        Returns:
            If legal move: 1
            If illegal move: -1
        """
        if self.model[edge_num] == 1:
            return -1
        else:
            self.model[edge_num] = 0
            return 1


    def add_edge(self,edge_num):
        """Change the model to reflect adding an edge.

        Set model[edge_num] to 1 if legal to do so (edge hasn't been set to 0 yet).

        Returns:
            If legal move: 1
            If illegal move: -1
        """
        if self.model[edge_num] == 0:
            return -1
        else:
            self.model[edge_num] = 1
            return 1

def euclidean(pt1,pt2):
    return math.sqrt(math.pow(pt1[0]-pt2[0],2) + math.pow(pt1[1]-pt2[1],2))

def filter_bad_new(copy,bad_fn,rows,cols,edge2index,man):
    """Create a dataset from the file that consists of only models that are consistent with the sdd
    If there is a file that already contains the indices of the bad paths, then don't recompute.
    If there is no such file, find the bad paths and store their indices in the file with name bad_fn.
    '../datasets/first_last-%d-%d-%d-%d-%d' % (rows,cols,start,end,run)
    """
    bad_lines = None
    bad_paths = {}

    """
    trip_id2bad_fn = 'pickles/trip_id2bad-%d-%d.pickle' % (rows,cols)
    file_exists = os.path.isfile(trip_id2bad_fn)
    if file_exists:
        trip_id2bad = pickle.load(open(trip_id2bad_fn,'rb'))
        for trip_id in trip_id2bad:
            bad_paths[trip_id-1] = True
    file_exists = os.path.isfile(bad_fn)
    if file_exists:
        bad_file = open(bad_fn,'r')
        bad_lines = bad_file.readlines()
        bad_file.close()
        for i in bad_lines:
            bad_paths[int(i)] = True
    """
    data = []

    copy.uniform_weights()
    bad_models = {}
    unique_bad = 0
    total_bad = 0
    total_good = 0
    unique_good = 0
    full_tuple = []

    bad_indices = {}
    good_models = {}
    bad_printed = 0
    times_printed = 1
    good_printed = 0
    file_exists = False

    t2model = pickle.load(open('../pickles/trip_id2model_better.pickle','rb'))
    for t in t2model:
        full_tuple.append(tuple(t2model[t]))

    if not file_exists: 
        cur_time = time.time()
        for i in range(len(full_tuple)):
            prev_time = cur_time
            cur_time = time.time()
            if times_printed < 100:
                print "time to evaluate model %d: %f" % (i-1,cur_time-prev_time)
            model = full_tuple[i]
            if str(model) in good_models:
                total_good += 1
                data.append(model)
                continue
            if str(model) in bad_models:
                total_bad += 1
                bad_indices[i] = True
                continue
            evidence = DataSet.evidence(model)
            probability = copy.probability(evidence)
            if probability == 0:
                if bad_printed < 25:
                    bad_printed += 1
                    print "Bad model:"
                    man.draw_grid(model)
                bad_models[str(model)] = True
                unique_bad += 1
                total_bad += 1
                bad_indices[i] = True
                continue

            else:
                if good_printed < 25:
                    good_printed += 1
                    print "Good model:"
                    #draw_grid(model,rows,cols,edge2index)
                good_models[str(model)] = True
                unique_good += 1
                total_good += 1
                data.append(model)

        print "total bad: %d, unique bad: %d, total good: %d, unique good: %d" % (total_bad,unique_bad, total_good, unique_good)

        bad_fn = 'bad_new.txt'
        bad_file = open(bad_fn,'w')
        for i in bad_indices.keys():
            bad_file.write("%d\n" % i)
        bad_file.close()

        full_dataset = DataSet.to_dict(data)

        return full_dataset

    else:
        for i in range(len(full_tuple)):
            model = full_tuple[i]
            if i in bad_paths:
                bad_models[str(model)] = True
                unique_bad += 1
                total_bad += 1
                continue

            else:
                if str(model) in good_models:
                    total_good += 1
                    data.append(model)
                    continue
                else:
                    good_models[str(model)] = True
                    unique_good += 1
                    total_good += 1
                    data.append(model)

        print "total bad: %d, unique bad: %d, total good: %d, unique good: %d" % (total_bad,unique_bad, total_good, unique_good)
        full_dataset = DataSet.to_dict(data)

        print "num bad paths: %d" % len(bad_paths)
        return full_dataset

def filter_bad(copy,in_fn,bad_fn,rows,cols,edge2index):
    """Create a dataset from the file that consists of only models that are consistent with the sdd
    If there is a file that already contains the indices of the bad paths, then don't recompute.
    If there is no such file, find the bad paths and store their indices in the file with name bad_fn.
    '../datasets/first_last-%d-%d-%d-%d-%d' % (rows,cols,start,end,run)
    """

    full_file = open(in_fn, "r")
    full_lines = full_file.readlines()
    full_file.close()

    full_ints = map(lambda x: map(int,x[:-1].split(',')),full_lines)
    full_tuple = map(tuple,full_ints)

    bad_lines = None
    bad_paths = {}

    trip_id2bad_fn = 'pickles/trip_id2bad-%d-%d.pickle' % (rows,cols)
    file_exists = os.path.isfile(trip_id2bad_fn)
    if file_exists:
        trip_id2bad = pickle.load(open(trip_id2bad_fn,'rb'))
        for trip_id in trip_id2bad:
            bad_paths[trip_id-1] = True
    """
    file_exists = os.path.isfile(bad_fn)
    if file_exists:
        bad_file = open(bad_fn,'r')
        bad_lines = bad_file.readlines()
        bad_file.close()
        for i in bad_lines:
            bad_paths[int(i)] = True
    """
    data = []

    copy.uniform_weights()
    bad_models = {}
    unique_bad = 0
    total_bad = 0
    total_good = 0
    unique_good = 0

    bad_indices = {}
    good_models = {}
    bad_printed = 0
    times_printed = 0
    good_printed = 0

    if not file_exists: 
        cur_time = time.time()
        for i in range(len(full_tuple)):
            prev_time = cur_time
            cur_time = time.time()
            #if times_printed < 100:
            #    print "time to evaluate model %d: %f" % (i-1,cur_time-prev_time)
            model = full_tuple[i]
            if str(model) in good_models:
                total_good += 1
                data.append(model)
                continue
            if str(model) in bad_models:
                total_bad += 1
                bad_indices[i] = True
                continue
            evidence = DataSet.evidence(model)
            probability = copy.probability(evidence)
            if probability == 0:
                if bad_printed < 25:
                    bad_printed += 1
                    #print "Bad model:"
                    #draw_grid(model,rows,cols,edge2index)
                bad_models[str(model)] = True
                unique_bad += 1
                total_bad += 1
                bad_indices[i] = True
                continue

            else:
                if good_printed < 25:
                    good_printed += 1
                    print "Good model:"
                    #draw_grid(model,rows,cols,edge2index)
                good_models[str(model)] = True
                unique_good += 1
                total_good += 1
                data.append(model)

        print "total bad: %d, unique bad: %d, total good: %d, unique good: %d" % (total_bad,unique_bad, total_good, unique_good)

        bad_file = open(bad_fn,'w')
        for i in bad_indices.keys():
            bad_file.write("%d\n" % i)
        bad_file.close()

        full_dataset = DataSet.to_dict(data)

        return full_dataset

    else:
        for i in range(len(full_tuple)):
            model = full_tuple[i]
            if i in bad_paths:
                bad_models[str(model)] = True
                unique_bad += 1
                total_bad += 1
                continue

            else:
                if str(model) in good_models:
                    total_good += 1
                    data.append(model)
                    continue
                else:
                    good_models[str(model)] = True
                    unique_good += 1
                    total_good += 1
                    data.append(model)

        print "total bad: %d, unique bad: %d, total good: %d, unique good: %d" % (total_bad,unique_bad, total_good, unique_good)
        full_dataset = DataSet.to_dict(data)

        print "num bad paths: %d" % len(bad_paths)
        return full_dataset

def generate_copy_new(fn_prefix):
    t2model = pickle.load(open('better_pickles/trip_id2model.pickle','rb'))
    t2in = pickle.load(open('better_pickles/t2training.pickle','rb'))

    models = []
    
    for t in t2in:
        models.append(tuple(t2model[t]))
    training = DataSet.to_dict(models)

    print training.N

    vtree_filename = '%s.vtree' % fn_prefix
    sdd_filename = '%s.sdd' % fn_prefix

    psi,scale = 2.0,None # learning hyper-parameters
    N,M = 2**10,2**10 # size of training/testing dataset
    em_max_iters = 10 # maximum # of iterations for EM
    em_threshold = 1e-4 # convergence threshold
    seed = 1 # seed for simulating datasets

    ########################################
    # READ INPUT
    ########################################

    print "== reading vtree/sdd"

    vtree = Vtree.read(vtree_filename)
    manager = SddManager(vtree)
    sdd = SddNode.read(sdd_filename,manager)
    pmanager = PSddManager(vtree)
    copy = pmanager.copy_and_normalize_sdd(sdd,vtree)
    pmanager.make_unique_true_sdds(copy,make_true=False) #AC: set or not set?

    psdd_parameters = copy.theta_count()




    print "Starting Training"
    start_time = time.time()
    copy.learn(training,psi=psi,scale=scale,show_progress=True)
    print "WORKED FOR TRIPS"
    print "== TRAINING =="
    print "    training time: %.3fs" % (time.time()-start_time)
    ll = copy.log_likelihood_alt(training)
    lprior = copy.log_prior(psi=psi,scale=scale)
    print "   training: %d unique, %d instances" % (len(training),training.N)
    print "   log likelihood: %.8f" % (ll/training.N)
    print "   log prior: %.8f" % (lprior/training.N)
    print "   log posterior: %.8f" % ((ll+lprior)/training.N)
    print "   log likelihood unnormalized: %.8f" % ll
    print "   log prior unnormalized: %.8f" % lprior
    print "   log posterior unnormalized: %.8f" % (ll+lprior)
    print "   log prior over parameters: %.8f" % (lprior/psdd_parameters)

    print "  zero parameters: %d (should be zero)" % copy.zero_count()
    copy.marginals()
    return copy

def gen_copy(rows,cols,fn_prefix):
    """Generate an untrained PSDD"""
    vtree_filename = '%s.vtree' % fn_prefix
    sdd_filename = '%s.sdd' % fn_prefix

    psi,scale = 2.0,None # learning hyper-parameters
    N,M = 2**10,2**10 # size of training/testing dataset
    em_max_iters = 10 # maximum # of iterations for EM
    em_threshold = 1e-4 # convergence threshold
    seed = 1 # seed for simulating datasets

    ########################################
    # READ INPUT
    ########################################

    print "== reading vtree/sdd"

    vtree = Vtree.read(vtree_filename)
    manager = SddManager(vtree)
    sdd = SddNode.read(sdd_filename,manager)
    pmanager = PSddManager(vtree)
    copy = pmanager.copy_and_normalize_sdd(sdd,vtree)
    pmanager.make_unique_true_sdds(copy,make_true=False) #AC: set or not set?
    return copy

    

def generate_copy(rows,cols,start,end,fn_prefix,data_fn,bad_fn,edge2index,num_edges):
    vtree_filename = '%s.vtree' % fn_prefix
    dd_filename = '%s.sdd' % fn_prefix

    psi,scale = 2.0,None # learning hyper-parameters
    N,M = 2**10,2**10 # size of training/testing dataset
    em_max_iters = 10 # maximum # of iterations for EM
    em_threshold = 1e-4 # convergence threshold
    seed = 1 # seed for simulating datasets

    ########################################
    # READ INPUT
    ########################################

    print "== reading vtree/sdd"

    vtree = Vtree.read(vtree_filename)
    manager = SddManager(vtree)
    sdd = SddNode.read(sdd_filename,manager)
    pmanager = PSddManager(vtree)
    copy = pmanager.copy_and_normalize_sdd(sdd,vtree)
    pmanager.make_unique_true_sdds(copy,make_true=False) #AC: set or not set?

    psdd_parameters = copy.theta_count()

    training = filter_bad(copy,data_fn,bad_fn,rows,cols,edge2index)

    start_time = time.time()
    copy.learn(training,psi=psi,scale=scale,show_progress=True)
    print "== TRAINING =="
    print "    training time: %.3fs" % (time.time()-start_time)
    ll = copy.log_likelihood_alt(training)
    lprior = copy.log_prior(psi=psi,scale=scale)
    print "   training: %d unique, %d instances" % (len(training),training.N)
    print "   log likelihood: %.8f" % (ll/training.N)
    print "   log prior: %.8f" % (lprior/training.N)
    print "   log posterior: %.8f" % ((ll+lprior)/training.N)
    print "   log likelihood unnormalized: %.8f" % ll
    print "   log prior unnormalized: %.8f" % lprior
    print "   log posterior unnormalized: %.8f" % (ll+lprior)
    print "   log prior over parameters: %.8f" % (lprior/psdd_parameters)

    print "  zero parameters: %d (should be zero)" % copy.zero_count()
    copy.marginals()
    return copy

def perform_analysis(rows,cols,start,end,fn_prefix,data_fn,bad_fn,edge2index,num_edges,trial_name):
    vtree_filename = '%s.vtree' % fn_prefix
    sdd_filename = '%s.sdd' % fn_prefix

    psi,scale = 2.0,None # learning hyper-parameters
    N,M = 2**10,2**10 # size of training/testing dataset
    em_max_iters = 10 # maximum # of iterations for EM
    em_threshold = 1e-4 # convergence threshold
    seed = 1 # seed for simulating datasets

    ########################################
    # READ INPUT
    ########################################

    print "== reading vtree/sdd"

    vtree = Vtree.read(vtree_filename)
    manager = SddManager(vtree)
    sdd = SddNode.read(sdd_filename,manager)
    pmanager = PSddManager(vtree)
    copy = pmanager.copy_and_normalize_sdd(sdd,vtree)
    pmanager.make_unique_true_sdds(copy,make_true=False) #AC: set or not set?

    psdd_parameters = copy.theta_count()

    training = filter_bad(copy,data_fn,bad_fn,rows,cols,edge2index)

    start_time = time.time()
    copy.learn(training,psi=psi,scale=scale,show_progress=True)
    print "== TRAINING =="
    print "    training time: %.3fs" % (time.time()-start_time)
    ll = copy.log_likelihood_alt(training)
    lprior = copy.log_prior(psi=psi,scale=scale)
    print "   training: %d unique, %d instances" % (len(training),training.N)
    print "   log likelihood: %.8f" % (ll/training.N)
    print "   log prior: %.8f" % (lprior/training.N)
    print "   log posterior: %.8f" % ((ll+lprior)/training.N)
    print "   log likelihood unnormalized: %.8f" % ll
    print "   log prior unnormalized: %.8f" % lprior
    print "   log posterior unnormalized: %.8f" % (ll+lprior)
    print "   log prior over parameters: %.8f" % (lprior/psdd_parameters)

    print "  zero parameters: %d (should be zero)" % copy.zero_count()
    copy.marginals()

    path_man = PathManager(rows,cols,edge2index,copy)

    print "%s PROBABILITIES" % trial_name
    path_man.visualize_mid_probs(start,end)

def find_kl(rows,cols,fn_prefix,bad_fn,data_fn):
    """Find the kl divergence for psdds trained on data taken from different time classes.

    Only run after determining the bad instances for the grid size.
    trip_id2class is 1-indexed while data lines and bad paths are 0-indexed
    """
    trip_id2class = pickle.load(open('../pickles/trip_id2class.pickle','rb'))
    with open(bad_fn,'r') as infile:
        bads = map(int,infile.readlines())
    bads.sort()
    bad_i = 0

    full_file = open(data_fn, "r")
    full_lines = full_file.readlines()
    full_file.close()

    r2
    full_ints = map(lambda x: map(int,x[:-1].split(',')),full_lines)
    full_tuples = map(tuple,full_ints)

    class_sets = []
    for i in range(6):
        class_sets.append([])

    for i in range(len(full_tuples)):
        if i == bads[bad_i]:
            bad_i += 1
            continue
        class_sets[trip_id2class[i+1]].append(full_tuples[i])

    class_sets = map(lambda x: DataSet.to_dict(x),class_sets)
    
    vtree_filename = '%s.vtree' % fn_prefix
    sdd_filename = '%s.sdd' % fn_prefix

    psi,scale = 2.0,None # learning hyper-parameters
    N,M = 2**10,2**10 # size of training/testing dataset
    em_max_iters = 10 # maximum # of iterations for EM
    em_threshold = 1e-4 # convergence threshold
    seed = 1 # seed for simulating datasets

    ########################################
    # READ INPUT
    ########################################

    print "== reading vtree/sdd"

    vtree = Vtree.read(vtree_filename)
    manager = SddManager(vtree)
    sdd = SddNode.read(sdd_filename,manager)

    psdds = []
    pmanagers = []
    for class_num in range(6):
        pmanager = PSddManager(vtree)
        copy = pmanager.copy_and_normalize_sdd(sdd,vtree)
        pmanager.make_unique_true_sdds(copy,make_true=False) #AC: set or not set?
        psdd_parameters = copy.theta_count()

        training = class_sets[class_num]

        start = time.time()
        copy.learn(training,psi=psi,scale=scale,show_progress=True)
        print "== TRAINING =="
        print "    training time: %.3fs" % (time.time()-start)
        ll = copy.log_likelihood_alt(training)
        lprior = copy.log_prior(psi=psi,scale=scale)
        print "   training: %d unique, %d instances" % (len(training),training.N)
        print "   log likelihood: %.8f" % (ll/training.N)
        print "   log prior: %.8f" % (lprior/training.N)
        print "   log posterior: %.8f" % ((ll+lprior)/training.N)
        print "   log likelihood unnormalized: %.8f" % ll
        print "   log prior unnormalized: %.8f" % lprior
        print "   log posterior unnormalized: %.8f" % (ll+lprior)
        print "   log prior over parameters: %.8f" % (lprior/psdd_parameters)

        print "  zero parameters: %d (should be zero)" % copy.zero_count()
        copy.marginals()

        print "\nCLASS %d HAS %d UNIQUE AND %d TOTAL INSTANCES\n" % (class_num,len(training),training.N)

        psdds.append(copy)
        pmanagers.append(pmanager)

    for i in range(6):
        for j in range(i+1,6):
            kl_divergence = PSddNode.kl_psdds(psdds[i],pmanagers[i],psdds[j],pmanagers[j])
            print "kl (%d,%d): %d" % (i,j,kl_divergence)
    
def edge_dsn(edge_path1,edge_path2):
    num_diff = 0.0
    for i in range(len(edge_path1)):
        if edge_path1[i] == 1 and edge_path2[i] != 1:
            num_diff += 1
        elif edge_path2[i] == 1 and edge_path1[i] != 1:
            num_diff += 1
    return num_diff/(edge_path1.count(1) + edge_path2.count(1))

def print_time_diff(start_time,op):
    time_dif = time.time() - start_time
    print "Time to compute %s: %.6f" % (op,time_dif)
    

def test_nearest_neighbor(rows,cols,edge2index,edge_index2tuple):
    man = PathManager(rows,cols,edge2index,edge_index2tuple)

    edge_path1 = [0 for i in range(man.num_edges)]
    edge_path2 = [0 for i in range(man.num_edges)]

    for i in (2,8,16,25,36):
        edge_path1[i] = 1
    for i in (6,14,24,33,43):
        edge_path2[i] = 1

    man.path_diff_measures(edge_path1,edge_path2)
    return
    haus,ampsd = man.min_and_sum_hausdorff(edge_path1,edge_path2)

    print "Hausdorff: %f" % haus
    print "Average Minimum Point Segment Distance: %f" % ampsd

    return

    nodes = [2,3,9,15,21,22,23]
    coords2in = {}
    for node in nodes:
        coords2in[man.node_to_tuple(node)] = True
    lookup_nodes = [3,4,6,14,21,29,26,31]
    for node in lookup_nodes:
        pt = man.node_to_tuple(node)
        nearest_dist = man.nearest_neighbor(pt,coords2in)
        print "Nearest neighbor to %d is %.4f" % (node,nearest_dist)

def exactly_one(edges,model):
    one_found = False
    for edge in edges:
        if model[edge] == 1:
            if one_found == True:
                return False
            one_found = True
    if one_found:
        return True
    else:
        return False

def most_frequent_model(model2ts):
    """Given a dict mapping models to a list of trip ids that have that model, return the model with the most trips."""
    best_score = 0
    best_model = None
    for model in model2ts:
        score = len(model2ts[model])
        if score > best_score:
            best_score = score
            best_model = model
    return best_model,best_score

def main():
    rows = int(sys.argv[1])
    cols = int(sys.argv[2])
    start = int(sys.argv[3])
    end = int(sys.argv[4])
    edge_filename = '../graphs/edge-nums-%d-%d.pickle' % (rows,cols)
    edge2index = pickle.load(open(edge_filename,'rb'))
    edge_tuple_filename = '../graphs/edge-to-tuple-%d-%d.pickle' % (rows,cols)
    edge_index2tuple = pickle.load(open(edge_tuple_filename,'rb'))
    num_edges = (rows-1)*cols + (cols-1)*rows

    #test_nearest_neighbor(rows,cols,edge2index,edge_index2tuple)

    #man = PathManager(rows,cols,edge2index)
    #man.start_set(start,end)
    #man.draw_all_paths()
    #return

    fn_prefix = '../graphs/general_ends-%d-%d' % (rows,cols)
    data_fn = '../datasets/general_ends-%d-%d.txt' % (rows,cols)
    bad_fn = 'bad_paths/general_bad-%d-%d.txt' % (rows,cols)
    fl2models_fn = 'better_pickles/first_last2models.pickle'
    testing_fl2models_fn = 'better_pickles/testing_fl2models.pickle' 
    fl2prediction_fn = 'better_pickles/fl2prediction.pickle'

    print "ENTIRE DATASET"
    man = PathManager(rows,cols,edge2index,edge_index2tuple,fl2models_fn=fl2models_fn,fl2prediction_fn=fl2prediction_fn)
    man.find_better_prediction()
    print "TESTING DATASET"
    man2 = PathManager(rows,cols,edge2index,edge_index2tuple,fl2models_fn=testing_fl2models_fn,fl2prediction_fn=fl2prediction_fn)
    man2.find_better_prediction()

    return
    man.number_guess_top_model()
    man.print_some_instances()
    man.visualize_similarities((start,end))
    #man.understand_similarity((start,end))
    return
    man.analyze_predictions_new()
    man.testing_fl2models()
    man.compare_observed_models_new()
    man.compare_observed_models()
    return
    copy = generate_copy_new(rows,cols,fn_prefix)
    copy = gen_copy(rows,cols,fn_prefix)
    filter_bad_new(copy,bad_fn,rows,cols,edge2index,man)
    return
    man.create_fl2models(data_fn,bad_fn)
    man = PathManager(rows,cols,edge2index,edge_index2tuple,copy)
    print "COPY GENERATED!!!"
    print man.best_all_at_once(5,84)
    return
 
    man.analyze_paths_taken()
    #find_kl(rows,cols,fn_prefix,bad_fn,data_fn)
    return


    copy = generate_copy(rows,cols,start,end,fn_prefix,data_fn,bad_fn,edge2index,num_edges)
    man = PathManager(rows,cols,edge2index,edge_index2tuple,copy)
    man.all_all_predictions()
    return
    s_time = time.time()
    all_prediction = man.best_all_at_once(start,end)
    print_time_diff(s_time,"all at once prediction")
    s_time = time.time()
    step_prediction = man.best_step_by_step(start,end)
    print_time_diff(s_time,"step by step prediction")


    print "STEP-BY-STEP PREDICTION"
    man.draw_grid(step_prediction)

    print "ALL-AT-ONCE PREDICTION"
    man.draw_grid(all_prediction)

    man.path_diff_measures(all_prediction,step_prediction)

    man.save_paths(start,end,step_prediction,all_prediction)
    return
    s_time = time.time()
    man.visualize_mid_probs(start,end)
    print_time_diff(s_time,"traversal probabilities")


if __name__ == '__main__':
    main()
