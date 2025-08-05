#################################
# Your name: Tomer Zalberg
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        xs = np.random.uniform(0, 1, m)
        xs = np.sort(xs)
        probs = np.where(((xs >= 0) & (xs <= 0.2)) | ((xs >= 0.4) & (xs <= 0.6)) | ((xs >= 0.8) & (xs <= 1.0)), 0.8, 0.1)
        ys = np.random.binomial(1, probs)
        return np.column_stack((xs, ys))
    
    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        results = []
        for n in range(m_first, m_last + 1, step):
            total_emp_error = 0.0
            total_true_error = 0.0
            for t in range(T):
                sample = self.sample_from_D(n)
                xs, ys = sample[:, 0], sample[:, 1]
                intervals_list, err = intervals.find_best_interval(xs, ys, k)
                true_error = self.true_error(intervals_list)
                empirical_error = (err / n)
                total_emp_error += empirical_error
                total_true_error += true_error
            avg_emp_error = total_emp_error / T
            avg_true_error = total_true_error / T
            results.append([avg_emp_error, avg_true_error])
        results = np.array(results)

        ns = np.arange(m_first, m_last + 1, step)

        plt.title("Experiment m range ERM")
        plt.xlabel("n")
        plt.plot(ns, results[:, 0], label="Empirical error")
        plt.plot(ns, results[:, 1], label="True error")
        plt.legend()
        plt.show()

        return results

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        results = []
        sample = self.sample_from_D(1500)
        xs, ys = sample[:, 0], sample[:, 1]
        for k in range(k_first, k_last + 1, step):
            intervals_list, err = intervals.find_best_interval(xs, ys, k)
            empirical_error = err / 1500
            true_error = self.true_error(intervals_list)
            results.append([empirical_error, true_error])

        results = np.array(results)
        ks = np.arange(k_first, k_last + 1, step)

        k_star_index = np.argmin(results[:, 0]) 
        k_star = ks[k_star_index]

        plt.title("Experiment k range ERM")
        plt.xlabel("k")
        plt.plot(ks, results[:, 0], label="Empirical error")
        plt.plot(ks, results[:, 1], label="True error")
        plt.legend()
        plt.show()

        return k_star

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        sample = self.sample_from_D(m)
        results = []
        np.random.shuffle(sample)

        split_size = int(0.8 * m)
        train_set = sample[:split_size]
        val_set = sample[split_size:]       

        train_set = train_set[train_set[:, 0].argsort()] 
        xs, ys = train_set[:, 0], train_set[:, 1]

        for k in range(1, 11):
            intervals_list, err = intervals.find_best_interval(xs, ys, k)
            val_error = self.empirical_error(intervals_list, val_set)
            results.append([k, val_error])

        results = np.array(results)
        k_star_index = np.argmin(results[:, 1])
        k_star = int(results[k_star_index, 0])

        return k_star



    #################################
    # Place for additional methods  

    def true_error(self, intervals_list):

        def p_y_given_x(x):
                if (0 <= x <= 0.2) or (0.4 <= x <= 0.6) or (0.8 <= x <= 1):
                    return 0.8
                else:
                    return 0.1
                
        def get_complement(intervals):
            intervals = sorted(intervals)
            comp = []
            prev = 0.0
            for (l, u) in intervals:
                if l > prev:
                    comp.append((prev, l))
                prev = max(prev, u)
            if prev < 1.0:
                comp.append((prev, 1.0))
            return comp
        
        def integrate_interval(a, b, label):
            
            # label = 1 means h(x)=1 - use 1 - P[y=1|x]
            # label = 0 means h(x)=0 - use P[y=1|x]
            total = 0.0
            boundaries = [a, b, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            boundaries = sorted(set(boundaries))
            for i in range(len(boundaries) - 1):
                l, r = boundaries[i], boundaries[i+1]
                if l >= b or r <= a:
                    continue
                sub_l = max(a, l)
                sub_r = min(b, r)
                mid = (sub_l + sub_r) / 2
                p = p_y_given_x(mid)
                if label == 1:
                    total += (sub_r - sub_l) * (1 - p)
                else:
                    total += (sub_r - sub_l) * p
            return total

        total_error = 0.0

        # error where h(x) = 1 (inside intervals)
        for l, u in intervals_list:
            total_error += integrate_interval(l, u, label=1)

        # error where h(x) = 0 (outside intervals)
        for l, u in get_complement(intervals_list):
            total_error += integrate_interval(l, u, label=0)

        return total_error


    def empirical_error(self, intervals_list, sample): 
        errors = 0
        for x, y in sample:
            h = any (l <= x <= u for (l, u) in intervals_list)
            if h != y:
                errors += 1
        return errors / len(sample)
    
    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.cross_validation(1500)


