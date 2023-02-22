/*
 * K Nearest Neighbors
 *
 * This algorithm calculates the euclidian distance
 * between the points in a training set, and the points 
 * in a query set, then makes a prediction by choosing the
 * mean (regression) or mode (classification) of the 
 * K nearest points from the query set.
 * */
use ndarray::prelude::*;
use ndarray::stack;

pub struct KNN {
    dataset : Array2<f64>
}


impl KNN {
    pub fn new(dataset : &Array2<f64>) -> Self {
        Self { dataset: dataset.clone() }
    }

    /* Calculates the distance between two N-dimensional points */
    /* The 'points' parameter should look like this:
     * [
     *      [x, y, z, w, ...],
     *      [x, y, z, w, ...]
     * ]
     * */
    fn eucledian_distance(points : &ArrayView2<f64>) -> f64 {
        let mut distance : f64 = 0.0;
        let point0 = points.slice(s![0,..]);
        let point1 = points.slice(s![1,..]);

        for i in 0..points.shape()[1] {
            distance += (point0[[i]] + point1[[i]]).powf(2.0);
        }

        distance = distance.sqrt();

        distance
    }


    pub fn mean(&self, point: &Array1<f64>, k : usize) -> Array1<f64> {
        let mut sorted_indices : Vec<(usize, f64)> = Vec::new(); // sorted indices of the closest neighbors
        let shape = self.dataset.shape();
        for i in 0..shape[0] {
            // Get the distance between the current point of the dataset and the query point
            let dataset_point = self.dataset.slice(s![i,..]);
            let points = stack![Axis(0), dataset_point, point.clone()];
            let points_view = points.view();
            let distance = Self::eucledian_distance(&points_view);
            sorted_indices.push((i, distance)); // store index of the dataset point and the distance
        }

        // sort neighbor indices based on distance
        sorted_indices.sort_by(|a, b| (a.1).partial_cmp(&b.1).unwrap());
        println!("Sorted Indices: {:?}", sorted_indices);

        // get mean value from the K nearest neighbors
        let mut mean_arr : Array1<f64> = Array1::zeros((point.len(),));
        for i in 0..k {
            let dataset_point = self.dataset.slice(s![sorted_indices[i].0,..]);
            for j in 0..mean_arr.len() {
                mean_arr[[j]] += dataset_point[[j]];
            }
        }

        for i in 0..mean_arr.len() {
            mean_arr[[i]] /= k as f64;
        }

        mean_arr
    }
}


