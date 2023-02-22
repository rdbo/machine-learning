/*
 * Machine Learning from Scratch
 */

mod knn; // K Nearest Neighbors

use ndarray::prelude::*;
use knn::*;

fn main() {
    let dataset : Array<f64, _> = array![
        [10.2, 25.0],
        [15.4, 32.9],
        [12.1, 57.4],
        [35.5, 32.4],
        [87.5, 69.0]
    ];

    let knn_data = KNN::new(&dataset);

    println!("Dataset: {}", dataset);
    println!("KNN Mode (k=3): {}", knn_data.mean(&array![12.1, 27.5], 3));
}
