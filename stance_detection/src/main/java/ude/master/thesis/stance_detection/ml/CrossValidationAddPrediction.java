package ude.master.thesis.stance_detection.ml;

/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    CrossValidationAddPrediction.java
 *    Copyright (C) 2009 University of Waikato, Hamilton, New Zealand
 *
 */

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AddClassification;

import java.io.IOException;
import java.util.Random;

/**
 * Performs a single run of cross-validation and adds the prediction on the test
 * set to the dataset.
 *
 * Command-line parameters:
 * <ul>
 * <li>-t filename - the dataset to use</li>
 * <li>-o filename - the output file to store dataset with the predictions in
 * </li>
 * <li>-x int - the number of folds to use</li>
 * <li>-s int - the seed for the random number generator</li>
 * <li>-c int - the class index, "first" and "last" are accepted as well; "last"
 * is used by default</li>
 * <li>-W classifier - classname and options, enclosed by double quotes; the
 * classifier to cross-validate</li>
 * </ul>
 *
 * Example command-line:
 * 
 * <pre>
 * java wekaexamples.classifiers.CrossValidationAddPrediction -t anneal.arff -c last -o predictions.arff -x 10 -s 1 -W "weka.classifiers.trees.J48 -C 0.25"
 * </pre>
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 5937 $
 */
public class CrossValidationAddPrediction {

	/**
	 * Performs the cross-validation. See Javadoc of class for information on
	 * command-line parameters.
	 *
	 * @param args
	 *            the command-line parameters
	 * @throws Exception
	 *             if something goes wrong
	 */

	private Classifier classifier;
	private int seed;
	private int folds;

	private Instances trainingdata;
	private String trainingdataLocation;
	private String trainClassIndex;

	private Instances testdata;
	private String testdataLocation;
	private String testClassIndex;

	public static void main(String[] args) throws Exception {
		// loads data and set class index
		CrossValidationAddPrediction cv = new CrossValidationAddPrediction();
		// cv.run(args, cv);
	}

	public Instances getTestdata() {
		return testdata;
	}

	public void setTestdata(Instances testdata) {
		this.testdata = testdata;
	}

	public String getTestdataLocation() {
		return testdataLocation;
	}

	public void setTestdataLocation(String testdataLocation) {
		this.testdataLocation = testdataLocation;
	}

	public String getTestClassIndex() {
		return testClassIndex;
	}

	public void setTestClassIndex(String testClassIndex) {
		this.testClassIndex = testClassIndex;
	}

	public void run(String trainingDataLoc, String traincClsIndex, int seed, int folds, Classifier cls,
			String testDataLoc, String testClsIndex) throws Exception, IOException {
		
		setTrainingdataLocation(trainingDataLoc);
		setClassIndex(traincClsIndex);
		loadTrainingSet(getTrainingdataLocation(), getTrainClassIndex());
		
		setTestdataLocation(testDataLoc);
		setTestClassIndex(testClsIndex);
		loadTrainingSet(getTestdataLocation(), getTestClassIndex());

		setSeed(seed);
		setFolds(folds);
		setClassifier(cls);

		
		// randomize data
		Random rand = new Random(seed);
		Instances randData = new Instances(trainingdata);
		randData.randomize(rand);
		if (randData.classAttribute().isNominal())
			randData.stratify(folds);

		// perform cross-validation and add predictions
		Instances predictedData = null;
		Evaluation eval = new Evaluation(randData);
		for (int n = 0; n < folds; n++) {
			// Instances train = randData.trainCV(folds, n);

			// Instances test = randData.testCV(folds, n);
			// the above code is used by the StratifiedRemoveFolds filter, the
			// code below by the Explorer/Experimenter:
			Instances train = randData.trainCV(folds, n, rand);

			// build and evaluate classifier
			Classifier clsCopy = AbstractClassifier.makeCopy(cls);
			clsCopy.buildClassifier(train);
			eval.evaluateModel(clsCopy, testdata);

		}

		// output evaluation
		System.out.println();
		System.out.println("=== Setup ===");
		if (cls instanceof OptionHandler)
			System.out.println("Classifier: " + cls.getClass().getName() + " "
					+ Utils.joinOptions(((OptionHandler) cls).getOptions()));
		else
			System.out.println("Classifier: " + cls.getClass().getName());
		System.out.println("Dataset: " + trainingdata.relationName());
		System.out.println("Folds: " + folds);
		System.out.println("Seed: " + seed);
		System.out.println();
		System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation ===", false));

		// output "enriched" dataset
		DataSink.write("resources/predicted0908.txt", predictedData);
	}

	public Classifier getClassifier() {
		return classifier;
	}

	public void setClassifier(Classifier classifier) {
		this.classifier = classifier;
	}

	public int getSeed() {
		return seed;
	}

	public void setSeed(int seed) {
		this.seed = seed;
	}

	public int getFolds() {
		return folds;
	}

	public void setFolds(int folds) {
		this.folds = folds;
	}

	public String getTrainClassIndex() {
		return trainClassIndex;
	}

	public void setClassIndex(String classIndex) {
		this.trainClassIndex = classIndex;
	}

	public String getTrainingdataLocation() {
		return trainingdataLocation;
	}

	public void setTrainingdataLocation(String trainingdataLocation) {
		this.trainingdataLocation = trainingdataLocation;
	}

	public void loadTrainingSet(String location, String clsIndex) throws Exception {
		Instances data = DataSource.read(location);
		if (clsIndex.length() == 0)
			clsIndex = "last";
		if (clsIndex.equals("first"))
			data.setClassIndex(0);
		else if (clsIndex.equals("last"))
			data.setClassIndex(data.numAttributes() - 1);
		else
			data.setClassIndex(Integer.parseInt(clsIndex) - 1);
		this.trainingdata = data;
	}
	
	public void loadTestSet(String location, String clsIndex) throws Exception {
		Instances data = DataSource.read(location);
		if (clsIndex.length() == 0)
			clsIndex = "last";
		if (clsIndex.equals("first"))
			data.setClassIndex(0);
		else if (clsIndex.equals("last"))
			data.setClassIndex(data.numAttributes() - 1);
		else
			data.setClassIndex(Integer.parseInt(clsIndex) - 1);
		this.testdata = data;
	}
}
