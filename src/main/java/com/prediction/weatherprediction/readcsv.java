/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.prediction.weatherprediction;

import org.apache.spark.sql.AnalysisException;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 *
 * @author wael abass
 */
public class readcsv {

    public static void main(String[] args) throws AnalysisException {
        SparkSession session = SparkSession.builder().appName("read csv").master("local[*]").config("spark.sql.warehouse.dir", "file:///E://").getOrCreate();
        // read csv using csv with header
        Dataset<Row> data = session.read().option("header", "true").csv("G:\\iris.csv");
        // display data using show method
        data.show();
        data.createTempView("csvdata");
        Dataset<Row> data1 = session.sql("select sepal_length from csvdata");
        //using select 
        Dataset<Row> data2 = data.select("sepal_length", "petal_length");
        data2.show();
    }

}
