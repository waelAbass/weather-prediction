/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.prediction.weatherprediction;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.sql.AnalysisException;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 *
 * @author wael
 */
public class readjsondata {

    public static void main(String[] args) throws AnalysisException {
        SparkSession session = SparkSession.builder().appName("play dataset").master("local[*]").config("spark.sql.warehouse.dir", "file:///E://").getOrCreate();
        Dataset<Row> data = session.read().json("G:\\ldata.json");
        data.show();
        data.createTempView("mynewdata");
        session.sql("select humidity from mynewdata ").show();
        Dataset<Row> newdata = data.select("humidity", "outlook");
        newdata.show();
        
        // select specific coulmns using select method 
        //Dataset<Row> select = data.select("humidity", "wind");
        //select.show();
        // select specific coulmns using SQL queries using temp view
        // data.createTempView("jdata");
        //session.sql("select play from jdata").show();
        // using   from spark ML
        // first step using string indexer
        StringIndexerModel indexer1 = new StringIndexer().
                setInputCol("humidity").setOutputCol("ohumidity").fit(data);
        StringIndexerModel indexer2 = new StringIndexer().
                setInputCol("outlook").setOutputCol("ooutlook").fit(data);

        StringIndexerModel indexer3 = new StringIndexer().
                setInputCol("tempreature").setOutputCol("otempreature").fit(data);
        StringIndexerModel indexer4 = new StringIndexer().
                setInputCol("wind").setOutputCol("owind").fit(data);
        StringIndexerModel target = new StringIndexer().
                setInputCol("play").setOutputCol("oplay").fit(data);
        // using pipeline to commit transformation (String indexer)
        Pipeline p1 = new Pipeline().setStages(new PipelineStage[]{indexer1, indexer2, indexer3, indexer4,
            target});
        Dataset<Row> data1 = p1.fit(data).transform(data);
        data1.show();
        // using vector assembler 
        VectorAssembler va = new VectorAssembler().setInputCols(new String[]{"ohumidity", "ooutlook", "otempreature", "owind"}).setOutputCol("features");
        Dataset<Row> data2 = va.transform(data1);
        data2.show();
        // using vector indexer to decide which features are categorical and convert original values to category indices
        VectorIndexerModel vim = new VectorIndexer().
                setInputCol("features").setOutputCol("ifeatures").setMaxCategories(4).fit(data2);
        DecisionTreeClassifier dt = new DecisionTreeClassifier().setLabelCol("oplay").setFeaturesCol("ifeatures");
        // label to string  to view value of prediction 
        IndexToString in = new IndexToString().setInputCol("oplay").
                setOutputCol("tplay").setLabels(target.labels());
        // using  Pipeline to achieve vector indexing model , estimator = decition tree
        Pipeline p2 = new Pipeline()
                .setStages(new PipelineStage[]{vim, dt, in});
        // spliting to train test 
        Dataset<Row>[] alldata = data2.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> training = alldata[0];
        Dataset<Row> test = alldata[1];
// using pipelinemodel to train(fit) model 
        PipelineModel pm = p2.fit(training);
        // using pipelinemodel to test model 
        pm.transform(test).show();
    }
}
