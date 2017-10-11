package ude.master.thesis.stance_detection.ml;

import static org.simmetrics.builders.StringDistanceBuilder.with;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Logger;
import org.clapper.util.misc.FileHashMap;
import org.clapper.util.misc.ObjectExistsException;
import org.clapper.util.misc.VersionMismatchException;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.simmetrics.StringDistance;
import org.simmetrics.metrics.CosineSimilarity;
import org.simmetrics.simplifiers.Simplifiers;
import org.simmetrics.tokenizers.Tokenizers;

import edu.mit.jwi.IDictionary;
import ude.master.thesis.stance_detection.processor.FeatureExtractor;
import ude.master.thesis.stance_detection.processor.FeatureExtractorWithModifiedBL;
import ude.master.thesis.stance_detection.processor.HypernymSimilarity;
import ude.master.thesis.stance_detection.processor.HypernymSimilarityWithArguments;
import ude.master.thesis.stance_detection.processor.Lemmatizer;
import ude.master.thesis.stance_detection.processor.NegationFeaturesGenerator2;
import ude.master.thesis.stance_detection.processor.NegationFeaturesGeneratorWithArguments;
import ude.master.thesis.stance_detection.processor.RelatedUnrelatedFeatureGenerator;
import ude.master.thesis.stance_detection.processor.RelatedUnrelatedFeatureGeneratorWithArguments;
import ude.master.thesis.stance_detection.processor.SVOFeaturesGenerator2;
import ude.master.thesis.stance_detection.processor.SVOFeaturesGeneratorWithArguments;
import ude.master.thesis.stance_detection.processor.SimilarityFeatures;
import ude.master.thesis.stance_detection.processor.SimilarityFeaturesWithArguments;
import ude.master.thesis.stance_detection.processor.StanfordDependencyParser2;
import ude.master.thesis.stance_detection.processor.StanfordDependencyParserWithArguments;
import ude.master.thesis.stance_detection.processor.Word2VecDataGenerator2;
import ude.master.thesis.stance_detection.processor.Word2VecDataGeneratorWithArguments;
import ude.master.thesis.stance_detection.util.BodySummerizer2;
import ude.master.thesis.stance_detection.util.FNCConstants;
import ude.master.thesis.stance_detection.util.LeskGlossOverlaps;
import ude.master.thesis.stance_detection.util.PPDBProcessor2;
import ude.master.thesis.stance_detection.util.PPDBProcessorWithArguments;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;
import ude.master.thesis.stance_detection.wordembeddings.DocToVec;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

public class FeaturesOrganiserWithArguments {
	final static Logger logger = Logger.getLogger(FeaturesOrganiserWithArguments.class);
	// Data
	private List<List<String>> trainingStances;
	private List<List<String>> testStances;
	private Map<Integer, String> trainingIdBoyMap;
	private HashMap<Integer, String> testIdBoyMap;

	private HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap;
	private HashMap<Integer, Map<Integer, String>> testSummIdBoyMap;
	// Instances
	private Instances trainingInstances;
	private Instances testInstances;
	private Instances unlabeledTestInstances;

	// Used features indicator

	// Ferreira
	private boolean useRootDistFeature = false;
	private boolean usePPDBFeature = false;
	private boolean useSVOFeature = false;
	private boolean useNegFeature = false;
	private boolean useBodyBoWCounterFeature = false;
	private boolean useBodyBoWSummCounterFeature = false;
	private boolean useWord2VecAddSimilarity = false;
	private boolean useHypernymsSimilarity = false;

	// Added features
	private boolean useTitleAndBodyParagraphVecs = false;

	// Baseline
	private boolean useOverlapFeature = false;
	private boolean useRefutingFeatures = false;
	private boolean usePolarityFeatures = false;
	private boolean useBinaryCooccurraneFeatures = false;
	private boolean useBinaryCooccurraneStopFeatures = false; // Same as binary
	// co-occurrence but ignores stopwords.
	private boolean useCharGramsFeatures = false;
	private boolean useWordGramsFeatures = false;

	// Add
	private boolean useMetricCosineSimilarity = false;
	private boolean useleskOverlap = false;
	private boolean useTitleQuestionMark = false;
	private boolean useTitleLength = false;

	private static StringDistance cosSimMetric;

	// Settings
	private boolean useBinaryRelatedUnrelatedClasses = false;
	private boolean useRelatedClasses = false;
	private boolean useAllClasses = false;
	private boolean useUnlabeledTestset = false;

	// Features data
	// Training data features
	private FileHashMap<String, ArrayList<Double>> trainRootDist;
	private FileHashMap<String, double[]> trainPPDB;
	private FileHashMap<String, Integer> trainNeg;
	private FileHashMap<String, Double> trainW2VSim;
	private FileHashMap<String, ArrayList<Integer>> trainSVO;

	private FileHashMap<String, Double> trainWordOverlap;
	private FileHashMap<String, ArrayList<Integer>> trainCoOccuranceStop;
	private FileHashMap<String, ArrayList<Integer>> trainCgram;
	private FileHashMap<String, ArrayList<Integer>> trainNgram;
	private FileHashMap<String, Double> trainHypSimilarity;
	private FileHashMap<String, Double> trainStrMetricCosSim;
	private FileHashMap<String, Double> trainLeskOverlap;

	// Test data features
	private FileHashMap<String, ArrayList<Double>> testRootDist;
	private FileHashMap<String, double[]> testPPDB;
	private FileHashMap<String, Integer> testNeg;
	private FileHashMap<String, Double> testW2VSim;
	private FileHashMap<String, ArrayList<Integer>> testSVO;

	private FileHashMap<String, Double> testWordOverlap;
	private FileHashMap<String, ArrayList<Integer>> testCoOccuranceStop;
	private FileHashMap<String, ArrayList<Integer>> testCgram;
	private FileHashMap<String, ArrayList<Integer>> testNgram;
	private FileHashMap<String, Double> testHypSimilarity;
	private FileHashMap<String, Double> testStrMetricCosSim;
	private FileHashMap<String, Double> testLeskOverlap;

	private FileHashMap<String, String> titlesLemmas;
	private FileHashMap<Integer, String> bodiesLemmas;

	private String arffFilename;

	private DocToVec doc2vec;
	private List<String> stanceValues;
	private LeskGlossOverlaps lgo;

	/**
	 * @throws IOException
	 * @throws VersionMismatchException
	 * @throws ClassNotFoundException
	 * @throws ObjectExistsException
	 * @throws FileNotFoundException
	 *             *
	 * 
	 */
	public void initializeFeatures(boolean saveARFF) throws FileNotFoundException, ObjectExistsException,
			ClassNotFoundException, VersionMismatchException, IOException {
		long startTimeExtraction = System.currentTimeMillis();
		initFeat(saveARFF);
		long endTimeExtraction = System.currentTimeMillis();
		System.out.println((double) (endTimeExtraction - startTimeExtraction) / 1000 + "s Feature-Extraction");
		logger.info("\n Feature-Extraction: " + (double) (endTimeExtraction - startTimeExtraction) / 1000);
	}

	private void initFeat(boolean saveARFF) throws FileNotFoundException, ObjectExistsException, ClassNotFoundException,
			VersionMismatchException, IOException {

		/*
		 * if (useMetricCosineSimilarity) { initCosSimilarityMetric(); }
		 */

		if (useRootDistFeature) {
			trainRootDist = StanfordDependencyParserWithArguments
					.loadRootDistFeaturesAsHashFile(ProjectPaths.ROOT_DIST_FEATURE_ARG_TRAIN);
			testRootDist = StanfordDependencyParserWithArguments
					.loadRootDistFeaturesAsHashFile(ProjectPaths.ROOT_DIST_FEATURE_ARG_TEST);

		}

		if (useSVOFeature) {
			trainSVO = SVOFeaturesGeneratorWithArguments
					.loadSVOFeaturesAsHashFile(ProjectPaths.SUMMED_SVO_FEATURE_ARG_TRAIN);
			testSVO = SVOFeaturesGeneratorWithArguments
					.loadSVOFeaturesAsHashFile(ProjectPaths.SUMMED_SVO_FEATURE_ARG_TEST);
		}

		if (usePPDBFeature) {
			trainPPDB = PPDBProcessorWithArguments
					.loadPPDBFeaturesAsHashFiles(ProjectPaths.SPPDB_HUNG_FEATURE_ARG_TRAIN);
			testPPDB = PPDBProcessorWithArguments.loadPPDBFeaturesAsHashFiles(ProjectPaths.SPPDB_HUNG_FEATURE_ARG_TEST);

		}

		if (useNegFeature) {
			trainNeg = NegationFeaturesGeneratorWithArguments
					.loadNegFeaturesAsHashFile(ProjectPaths.NEG_FEATURE_ARG_TRAIN);
			testNeg = NegationFeaturesGeneratorWithArguments
					.loadNegFeaturesAsHashFile(ProjectPaths.NEG_FEATURE_ARG_TEST);
		}

		if (useWord2VecAddSimilarity) {
			trainW2VSim = Word2VecDataGeneratorWithArguments
					.loadWord2VecFeaturesAsHashFile(ProjectPaths.W2V_SIM_ADD_ARG_TRAIN);
			testW2VSim = Word2VecDataGeneratorWithArguments
					.loadWord2VecFeaturesAsHashFile(ProjectPaths.W2V_SIM_ADD_ARG_TEST);

		}

		if (useOverlapFeature) {
			trainWordOverlap = RelatedUnrelatedFeatureGeneratorWithArguments
					.loadWordsOverlapsFeaturesAsHashFile(ProjectPaths.TRAIN_ARG_WORD_OVERLAPS_PATH);
			testWordOverlap = RelatedUnrelatedFeatureGeneratorWithArguments
					.loadWordsOverlapsFeaturesAsHashFile(ProjectPaths.TEST_ARG_WORD_OVERLAPS_PATH);
		}

		if (useBinaryCooccurraneStopFeatures) {
			trainCoOccuranceStop = RelatedUnrelatedFeatureGeneratorWithArguments
					.loadCoOccStopFeaturesAsHashFile(ProjectPaths.TRAIN_ARG_COOCC_PATH);
			testCoOccuranceStop = RelatedUnrelatedFeatureGeneratorWithArguments
					.loadCoOccStopFeaturesAsHashFile(ProjectPaths.TEST_ARG_COOCC_PATH);
		}

		if (useCharGramsFeatures) {
			trainCgram = RelatedUnrelatedFeatureGeneratorWithArguments
					.loadCharGramsFeaturesAsHashFile(ProjectPaths.TRAIN_ARG_CGRAMS_PATH);
			testCgram = RelatedUnrelatedFeatureGeneratorWithArguments
					.loadCharGramsFeaturesAsHashFile(ProjectPaths.TEST_ARG_CGRAMS_PATH);
		}

		if (useWordGramsFeatures) {
			trainNgram = RelatedUnrelatedFeatureGeneratorWithArguments
					.loadNGramsFeaturesAsHashFile(ProjectPaths.TRAIN_ARG_NGRAMS_PATH);
			testNgram = RelatedUnrelatedFeatureGeneratorWithArguments
					.loadNGramsFeaturesAsHashFile(ProjectPaths.TEST_ARG_NGRAMS_PATH);
		}

		if (useMetricCosineSimilarity) {
			trainStrMetricCosSim = SimilarityFeaturesWithArguments
					.loadCosSimFeaturesAsHashFile(ProjectPaths.TRAIN_ARG_COS_SIM_STRMET_PATH);
			testStrMetricCosSim = SimilarityFeaturesWithArguments
					.loadCosSimFeaturesAsHashFile(ProjectPaths.TEST_ARG_COS_SIM_STRMET_PATH);
		}

		if (useHypernymsSimilarity) {
			trainHypSimilarity = HypernymSimilarityWithArguments
					.loadHypSimAsHashFile(ProjectPaths.TRAIN_ARG_HYP_SIM_PATH);
			testHypSimilarity = HypernymSimilarityWithArguments
					.loadHypSimAsHashFile(ProjectPaths.TEST_ARG_HYP_SIM_PATH);
		}

		if (useleskOverlap) {
			trainLeskOverlap = SimilarityFeaturesWithArguments
					.loadLeskFeaturesAsHashFile(ProjectPaths.TRAIN_ARG_LESK_PATH);
			testLeskOverlap = SimilarityFeaturesWithArguments
					.loadLeskFeaturesAsHashFile(ProjectPaths.TEST_ARG_LESK_PATH);
		}

		// Load Lemmatized data
		titlesLemmas = Lemmatizer.loadTitlesLemmasAsHashFiles(ProjectPaths.TITLES_LEMMAS);
		bodiesLemmas = Lemmatizer.loadBodiesAsWholeLemmasAsHashFiles(ProjectPaths.ARGS_BODIES_LEMMAS);

		trainingInstances = initializeInstances("fnc-1", trainingStances, trainingIdBoyMap, "");
		testInstances = initializeInstances("fnc-1", testStances, testIdBoyMap, "");

		if (useUnlabeledTestset)
			unlabeledTestInstances = initializeInstances("fnc-1", testStances, testIdBoyMap, "?");

		if (saveARFF) {
			StanceDetectionDataReader.saveInstancesToArff(arffFilename + "_train", trainingInstances);
			StanceDetectionDataReader.saveInstancesToArff(arffFilename + "_test", testInstances);

			if (useUnlabeledTestset)
				StanceDetectionDataReader.saveInstancesToArff(arffFilename + "_unlabeled_test", unlabeledTestInstances);
		}
	}

	private Instances initializeInstances(String string, List<List<String>> stances,
			Map<Integer, String> trainingIdBoyMap, String unlabeled) throws IOException {
		ArrayList<Attribute> features = new ArrayList<>();

		if (useBodyBoWCounterFeature) {
			features.add(new Attribute(FNCConstants.BODY_BOW_COUNTER, (List<String>) null));
		}
		if (useBodyBoWSummCounterFeature) {
			features.add(new Attribute(FNCConstants.BODY_SUMM_BOW_COUNTER, (List<String>) null));
		}
		if (useRootDistFeature) {
			for (int i = 0; i < 5; i++)
				features.add(new Attribute(FNCConstants.ROOT_DIST_REFUTE + i));

			for (int i = 5; i < 10; i++)
				features.add(new Attribute(FNCConstants.ROOT_DIST_DISCUSS + i));
		}

		if (usePPDBFeature) {
			features.add(new Attribute(FNCConstants.PPDB_HUNG));
		}

		if (useNegFeature) {
			features.add(new Attribute(FNCConstants.NEG_FEATURE));
		}

		if (useWord2VecAddSimilarity) {
			features.add(new Attribute(FNCConstants.WORD2VEC_ADD_SIM));
		}

		if (useSVOFeature) {
			for (int i = 0; i < 12; i++) {
				features.add(new Attribute(FNCConstants.SVO + i));
			}
		}

		if (useTitleAndBodyParagraphVecs) {
			for (int i = 0; i < 200; i++) {
				features.add(new Attribute(FNCConstants.TITLE_DOC2VEC + i));
			}
			for (int i = 0; i < 200; i++) {
				features.add(new Attribute(FNCConstants.BODY_DOC2VEC + i));
			}
		}

		if (useOverlapFeature) {
			features.add(new Attribute(FNCConstants.WORD_OVERLAP));
		}

		if (useRefutingFeatures) {
			for (String refute : FeatureExtractor.refutingWords) {
				features.add(new Attribute(FNCConstants.REFUTE_FEATURE + refute));
			}
		}

		if (usePolarityFeatures) {
			features.add(new Attribute(FNCConstants.TITLE_POLARITY));
			features.add(new Attribute(FNCConstants.BODY_POLARITY));
		}

		if (useBinaryCooccurraneFeatures) {
			features.add(new Attribute(FNCConstants.BINARY_COOCCURANCE_COUNT));
			features.add(new Attribute(FNCConstants.BINARY_COOCCURANCE_COUNT_FIRST_255)); // just
		}

		if (useBinaryCooccurraneStopFeatures) {
			for (int i = 0; i < 2; i++) {
				features.add(new Attribute(FNCConstants.BINARY_COOCCURANCE_STOP_COUNT + i));
			}
		}

		if (useCharGramsFeatures) {
			int featSize = 3 * 3;
			for (int i = 0; i < featSize; i++) {
				features.add(new Attribute(FNCConstants.CHAR_GRAMS_HITS + i));
			}
		}

		if (useWordGramsFeatures) {
			int featSize = 4 * 2;
			for (int i = 0; i < featSize; i++) {
				features.add(new Attribute(FNCConstants.NGRAM_HITS + i));
			}
		}

		if (useMetricCosineSimilarity) {
			features.add(new Attribute(FNCConstants.COSINE_METRIC_SIM));
		}

		if (useleskOverlap) {
			features.add(new Attribute(FNCConstants.LESK_OVERLAP));
		}

		if (useHypernymsSimilarity) {
			features.add(new Attribute(FNCConstants.HYP_SIM));
		}

		if (useTitleQuestionMark) {
			features.add(new Attribute(FNCConstants.TITLE_Q));
		}

		if (useTitleLength) {
			features.add(new Attribute(FNCConstants.TITLE_LENGTH));
		}

		// Add the classs attribute
		stanceValues = new ArrayList<>();
		if (useAllClasses) {
			stanceValues = Arrays.asList(FNCConstants.ALL_STANCE_CLASSES);
		} else if (useRelatedClasses) {
			stanceValues = Arrays.asList(FNCConstants.RELATED_STANCE_CLASSES);
		} else if (useBinaryRelatedUnrelatedClasses) {
			stanceValues = Arrays.asList(FNCConstants.BINARY_STANCE_CLASSES);
		}

		if (unlabeled.equals("?")) {
			stanceValues = Arrays.asList(FNCConstants.ALL_STANCE_CLASSES);
			features.add(new Attribute(FNCConstants.CLASS_ATTRIBUTE_NAME, stanceValues));
		} else
			features.add(new Attribute(FNCConstants.CLASS_ATTRIBUTE_NAME, stanceValues));

		Instances instances = new Instances(FNCConstants.RELATION_NAME, features, stances.size());

		instances.setClassIndex(features.size() - 1);

		assignInstancesValues(stances, trainingIdBoyMap, instances, unlabeled, features.size());
		return instances;
	}

	private void assignInstancesValues(List<List<String>> stances, Map<Integer, String> trainingIdBoyMap,
			Instances instances, String unlabeled, int featuresSize) throws IOException {
		System.out.println("Started getting instances");
		int i = 0;
		for (List<String> stance : stances) {
			// this if stmt is related to how many classes are used in testing
			// the classifier
			if (useBinaryRelatedUnrelatedClasses) {

				String headline = stance.get(0);
				/*
				 * String bodyPart1 =
				 * (bodiesLemmas.get(Integer.valueOf(stance.get(1))).get(1));
				 * String bodyPart2 =
				 * (bodiesLemmas.get(Integer.valueOf(stance.get(1))).get(3));
				 * 
				 * ArrayList<String> bodyParts = new ArrayList<>();
				 * bodyParts.add(bodyPart1); bodyParts.add(bodyPart2);
				 */

				String body = (bodiesLemmas.get(Integer.valueOf(stance.get(1))));

				DenseInstance instance = createInstance(headline, body, stance.get(1), instances, featuresSize);
				if (unlabeled.equals("?")) {
					instance.setClassMissing();
				} else {
					if (!stance.get(2).equals("unrelated"))
						instance.setClassValue("related");
					else
						instance.setClassValue(stance.get(2));
				}
				instances.add(instance);

			} else if (useRelatedClasses || useAllClasses) {
				if (stanceValues.contains(stance.get(2))) {
					String headline = stance.get(0);
					// String bodyPart1 = FeatureExtractorWithModifiedBL
					// .clean(bodiesLemmas.get(Integer.valueOf(stance.get(1))).get(1));
					// String bodyPart2 = FeatureExtractorWithModifiedBL
					// .clean(bodiesLemmas.get(Integer.valueOf(stance.get(1))).get(3));

					/*
					 * ArrayList<String> bodyParts = new ArrayList<>();
					 * bodyParts.add(bodyPart1); bodyParts.add(bodyPart2);
					 */

					String body = FeatureExtractorWithModifiedBL
							.getLemmatizedCleanStr(trainingIdBoyMap.get(Integer.valueOf(stance.get(1))));

					DenseInstance instance = createInstance(headline, body, stance.get(1), instances, featuresSize);
					// System.out.println(stance.get(2));
					if (unlabeled.equals("?")) {
						instance.setClassMissing();
					} else
						instance.setClassValue(stance.get(2));
					instances.add(instance);
				}
			}

			i++;
			if (i % 10000 == 0)
				System.out.println("Have read " + instances.size() + " instances");
		}

		System.out.println("Finished getting instances");

	}

	/***
	 * 
	 * @param headline
	 * @param body
	 * @param bodyId
	 * @param instances
	 * @param featuresSize
	 * @param instanceType
	 * @return
	 * @throws IOException
	 */
	private DenseInstance createInstance(String headline, String body, String bodyId, Instances instances,
			int featuresSize) throws IOException {
		DenseInstance instance = new DenseInstance(featuresSize);
		instance.setDataset(instances);

		if (useBodyBoWCounterFeature) { // we use this just for the related type
										// classification
			instance.setValue(instances.attribute(FNCConstants.BODY_BOW_COUNTER), body);
		}

		if (useBodyBoWSummCounterFeature) { // we use this just for the related type
			// classification
			String parts;
			if(trainingSummIdBoyMap.containsKey(Integer.valueOf(bodyId)))
				parts = trainingSummIdBoyMap.get(Integer.valueOf(bodyId)).get(1)+ " " +
						trainingSummIdBoyMap.get(Integer.valueOf(bodyId)).get(3);
			else{
				parts = testSummIdBoyMap.get(Integer.valueOf(bodyId)).get(1)+ " " +
						testSummIdBoyMap.get(Integer.valueOf(bodyId)).get(3);
			}
			instance.setValue(instances.attribute(FNCConstants.BODY_SUMM_BOW_COUNTER), parts);
		}

		if (useRootDistFeature) {
			ArrayList<Double> rootdist = null;

			rootdist = trainRootDist.get(bodyId);
			if (rootdist == null)
				rootdist = testRootDist.get(bodyId);

			for (int i = 0; i < 5; i++)
				instance.setValue(instances.attribute(FNCConstants.ROOT_DIST_REFUTE + i), rootdist.get(i));
			for (int i = 5; i < 10; i++)
				instance.setValue(instances.attribute(FNCConstants.ROOT_DIST_DISCUSS + i), rootdist.get(i));

		}

		if (usePPDBFeature) {
			double[] ppdb = null;

			ppdb = trainPPDB.get(headline + bodyId);
			if (ppdb == null)
				ppdb = testPPDB.get(headline + bodyId);

			instance.setValue(instances.attribute(FNCConstants.PPDB_HUNG), ppdb[ppdb.length - 1]);

		}

		if (useNegFeature) {
			Integer neg = null;

			neg = trainNeg.get(headline + bodyId);
			if (neg == null)
				neg = testNeg.get(headline + bodyId);

			instance.setValue(instances.attribute(FNCConstants.NEG_FEATURE), neg);
		}

		if (useWord2VecAddSimilarity) {
			Double sim = null;

			sim = trainW2VSim.get(headline + bodyId);
			if (sim == null)
				sim = testW2VSim.get(headline + bodyId);

			instance.setValue(instances.attribute(FNCConstants.WORD2VEC_ADD_SIM), sim);
		}

		if (useSVOFeature) {
			ArrayList<Integer> svos = null;

			svos = trainSVO.get(headline + bodyId);
			if (svos == null)
				svos = testSVO.get(headline + bodyId);

			// add svo features vector
			for (int i = 0; i < svos.size(); i++) {
				instance.setValue(instances.attribute(FNCConstants.SVO + i), svos.get(i));
			}
		}

		if (useTitleAndBodyParagraphVecs) {

			if (doc2vec == null)
				doc2vec = new DocToVec();

			double[] tVec = doc2vec.getTitleParagraphVecByLabel(headline);
			for (int i = 0; i < 100; i++) {
				instance.setValue(instances.attribute(FNCConstants.TITLE_DOC2VEC + i), tVec[i]);
			}

			double[] bVec = doc2vec.getBodyParagraphVecByLabel(bodyId);
			for (int i = 0; i < 100; i++) {
				instance.setValue(instances.attribute(FNCConstants.BODY_DOC2VEC + i), bVec[i]);
			}

		}

		if (useOverlapFeature) {
			Double wordOverlap = null;
			wordOverlap = trainWordOverlap.get(headline + bodyId);
			if (wordOverlap == null)
				wordOverlap = testWordOverlap.get(headline + bodyId);

			Attribute wordOverlapAtt = instances.attribute(FNCConstants.WORD_OVERLAP);
			instance.setValue(wordOverlapAtt, wordOverlap);
		}

		if (useRefutingFeatures) {
			for (String refute : FeatureExtractorWithModifiedBL.refutingWords) {
				Attribute refuteAtts = instances.attribute(FNCConstants.REFUTE_FEATURE + refute);
				instance.setValue(refuteAtts, FeatureExtractorWithModifiedBL.getRefutingFeature(headline, refute));
			}
		}

		// TODO: split to 2 features use
		if (usePolarityFeatures) {
			Attribute headPolarityAtt = instances.attribute(FNCConstants.TITLE_POLARITY);
			// String headlineLemma =
			// FeatureExtractor.getLemmatizedCleanStr(headline);
			String headlineLemma = FeatureExtractorWithModifiedBL.clean(titlesLemmas.get(headline));

			instance.setValue(headPolarityAtt, FeatureExtractorWithModifiedBL.calculatePolarity(headlineLemma));

			Attribute bodyPolarityAtt = instances.attribute(FNCConstants.BODY_POLARITY);
			instance.setValue(bodyPolarityAtt, FeatureExtractorWithModifiedBL.calculatePolarity(body));
		}
		// Not used
		if (useBinaryCooccurraneFeatures) {
			List<Integer> binCoOccFeats = FeatureExtractorWithModifiedBL.getBinaryCoOccurenceFeatures(headline, body);
			Attribute binCoOccAtt = instances.attribute(FNCConstants.BINARY_COOCCURANCE_COUNT);
			instance.setValue(binCoOccAtt, binCoOccFeats.get(0));

			Attribute binCoOccEarlyAtt = instances.attribute(FNCConstants.BINARY_COOCCURANCE_COUNT_FIRST_255);
			instance.setValue(binCoOccEarlyAtt, binCoOccFeats.get(1));
		}

		if (useBinaryCooccurraneStopFeatures) {
			ArrayList<Integer> coOccuranceStop = null;
			coOccuranceStop = trainCoOccuranceStop.get(headline + bodyId);
			if (coOccuranceStop == null)
				coOccuranceStop = testCoOccuranceStop.get(headline + bodyId);

			for (int i = 0; i < 2; i++) {
				Attribute coOccuranceStopAtt = instances.attribute(FNCConstants.BINARY_COOCCURANCE_STOP_COUNT + i);
				instance.setValue(coOccuranceStopAtt, coOccuranceStop.get(i));
			}
		}

		if (useCharGramsFeatures) {
			ArrayList<Integer> cgram = null;
			cgram = trainCgram.get(headline + bodyId);
			if (cgram == null)
				cgram = testCgram.get(headline + bodyId);

			int featSize = 3 * 3;

			for (int i = 0; i < featSize; i++) {
				Attribute cgramAtt = instances.attribute(FNCConstants.CHAR_GRAMS_HITS + i);
				instance.setValue(cgramAtt, cgram.get(i));
			}
		}

		if (useWordGramsFeatures) {
			int featSize = 4 * 2;

			ArrayList<Integer> ngram = null;
			ngram = trainNgram.get(headline + bodyId);
			if (ngram == null)
				ngram = testNgram.get(headline + bodyId);

			for (int i = 0; i < featSize; i++) {
				Attribute ngramAtt = instances.attribute(FNCConstants.NGRAM_HITS + i);
				instance.setValue(ngramAtt, ngram.get(i));
			}
		}

		if (useleskOverlap) {
			Double lesk = null;
			lesk = trainLeskOverlap.get(headline + bodyId);
			if (lesk == null)
				lesk = testLeskOverlap.get(headline + bodyId);

			Attribute ngramAtt = instances.attribute(FNCConstants.LESK_OVERLAP);
			instance.setValue(ngramAtt, lesk);
		}

		if (useTitleQuestionMark) {
			int q = 0;
			if (headline.contains("?"))
				q = 1;
			Attribute qAtt = instances.attribute(FNCConstants.TITLE_Q);
			instance.setValue(qAtt, q);
		}

		if (useTitleLength) {
			int len = headline.split("\\s+").length;
			Attribute lenAtt = instances.attribute(FNCConstants.TITLE_LENGTH);
			instance.setValue(lenAtt, len);
		}

		if (useMetricCosineSimilarity) {
			Double cos = null;
			cos = trainStrMetricCosSim.get(headline + bodyId);
			if (cos == null)
				cos = testStrMetricCosSim.get(headline + bodyId);

			Attribute cosAtt = instances.attribute(FNCConstants.COSINE_METRIC_SIM);
			instance.setValue(cosAtt, cos);
		}

		if (useHypernymsSimilarity) {
			Double hyp = null;
			hyp = trainHypSimilarity.get(headline + bodyId);
			if (hyp == null)
				hyp = testHypSimilarity.get(headline + bodyId);

			Attribute cosAtt = instances.attribute(FNCConstants.HYP_SIM);
			instance.setValue(cosAtt, hyp);
		}

		return instance;
	}

	public void loadData() throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.ARGUMENTED_BODIES_TRAIN,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.ARGUMENTED_BODIES_TEST);

		trainingIdBoyMap = sddr.getTrainIdBodyMap();
		testIdBoyMap = sddr.getTestIdBodyMap();
		System.out.println(trainingIdBoyMap.size());
		trainingStances = sddr.getTrainStances();
		System.out.println(trainingStances.size());

		trainingStances = removeInstanceswithEmptyBodies(trainingStances, trainingIdBoyMap);

		testStances = sddr.getTestStances();

		testStances = removeInstanceswithEmptyBodies(testStances, testIdBoyMap);

		trainingSummIdBoyMap = sddr.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TRAIN_BODIES));
		testSummIdBoyMap = sddr.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TEST_BODIES));
	}

	private List<List<String>> removeInstanceswithEmptyBodies(List<List<String>> stances,
			Map<Integer, String> idBoyMap) {
		System.out.println("Before: " + stances.size());
		List<List<String>> newstances = new ArrayList<>();
		for (List<String> s : stances) {
			Integer bodyId = Integer.valueOf(s.get(1));
			String body = idBoyMap.get(bodyId);
			if (!body.trim().isEmpty())
				newstances.add(s);
		}
		System.out.println("After: " + newstances.size());
		return newstances;
	}

	public boolean isUseRootDistFeature() {
		return useRootDistFeature;
	}

	public void useRootDistFeature(boolean useRootDistFeature) {
		this.useRootDistFeature = useRootDistFeature;
	}

	public boolean isPPDBFeature() {
		return usePPDBFeature;
	}

	public void usePPDBFeature(boolean usePPDBFeature) {
		this.usePPDBFeature = usePPDBFeature;
	}

	public boolean isWord2VecAddSimilarityUsed() {
		return useWord2VecAddSimilarity;
	}

	public void useWord2VecAddSimilarity(boolean useWord2VecSimilarity) {
		this.useWord2VecAddSimilarity = useWord2VecSimilarity;
	}

	public boolean isSVOFeatureUsed() {
		return useSVOFeature;
	}

	public void useSVOFeature(boolean useSVOFeature) {
		this.useSVOFeature = useSVOFeature;
	}

	public boolean isNegFeatureUsed() {
		return useNegFeature;
	}

	public void useNegFeature(boolean useNegFeature) {
		this.useNegFeature = useNegFeature;
	}

	public boolean isOverlapFeatureUsed() {
		return useOverlapFeature;
	}

	public void useOverlapFeature(boolean useOverlapFeature) {
		this.useOverlapFeature = useOverlapFeature;
	}

	public boolean isRefutingFeaturesUsed() {
		return useRefutingFeatures;
	}

	public void useRefutingFeatures(boolean useRefutingFeatures) {
		this.useRefutingFeatures = useRefutingFeatures;
	}

	public boolean isPolarityFeaturesUsed() {
		return usePolarityFeatures;
	}

	public void usePolarityFeatures(boolean usePolarityFeatures) {
		this.usePolarityFeatures = usePolarityFeatures;
	}

	public boolean isBinaryCooccurraneFeaturesUsed() {
		return useBinaryCooccurraneFeatures;
	}

	public void useBinaryCooccurraneFeatures(boolean useBinaryCooccurraneFeatures) {
		this.useBinaryCooccurraneFeatures = useBinaryCooccurraneFeatures;
	}

	public boolean isCharGramsFeaturesUsed() {
		return useCharGramsFeatures;
	}

	public void useCharGramsFeatures(boolean useCharGramsFeatures) {
		this.useCharGramsFeatures = useCharGramsFeatures;
	}

	public boolean isWordGramsFeaturesUsed() {
		return useWordGramsFeatures;
	}

	public void useWordGramsFeatures(boolean useWordGramsFeatures) {
		this.useWordGramsFeatures = useWordGramsFeatures;
	}

	public boolean isBinaryCooccurraneStopFeaturesUsed() {
		return useBinaryCooccurraneStopFeatures;
	}

	public void useBinaryCooccurraneStopFeatures(boolean useBinaryCooccurraneStopFeatures) {
		this.useBinaryCooccurraneStopFeatures = useBinaryCooccurraneStopFeatures;
	}

	public boolean isBodyBoWCounterFeatureUsed() {
		return useBodyBoWCounterFeature;
	}

	public void useBodyBoWCounterFeature(boolean useBodyBoWCounterFeature) {
		this.useBodyBoWCounterFeature = useBodyBoWCounterFeature;
	}

	public boolean isTitleAndBodyParagraphVecsUsed() {
		return useTitleAndBodyParagraphVecs;
	}

	public void useTitleAndBodyParagraphVecs(boolean useTitleAndBodyParagraphVecs) {
		this.useTitleAndBodyParagraphVecs = useTitleAndBodyParagraphVecs;
	}

	public boolean isMetricCosineSimilarityUsed() {
		return useMetricCosineSimilarity;
	}

	public void useMetricCosineSimilarity(boolean useMetricCosineSimilarity) {
		this.useMetricCosineSimilarity = useMetricCosineSimilarity;
	}

	public boolean isBinaryRelatedUnrelatedClassesUsed() {
		return useBinaryRelatedUnrelatedClasses;
	}

	public void useBinaryRelatedUnrelatedClasses(boolean useBinaryRelatedUnrelatedClasses) {
		this.useBinaryRelatedUnrelatedClasses = useBinaryRelatedUnrelatedClasses;
	}

	public boolean isRelatedClassesUsed() {
		return useRelatedClasses;
	}

	public void useRelatedClasses(boolean useRelatedClasses) {
		this.useRelatedClasses = useRelatedClasses;
	}

	public boolean isAllClassesUsed() {
		return useAllClasses;
	}

	public void useAllClasses(boolean useAllClasses) {
		this.useAllClasses = useAllClasses;
	}

	public String getArffFilename() {
		return arffFilename;
	}

	public void setArffFilename(String arffFilename) {
		this.arffFilename = arffFilename;
	}

	public Instances getTrainingInstances() {
		return trainingInstances;
	}

	public void setTrainingInstances(Instances trainingInstances) {
		this.trainingInstances = trainingInstances;
	}

	public Instances getTestInstances() {
		return testInstances;
	}

	public void setTestInstances(Instances testInstances) {
		this.testInstances = testInstances;
	}

	public boolean isUnlabeledTestsetUsee() {
		return useUnlabeledTestset;
	}

	public void useUnlabeledTestset(boolean useUnlabeledTestset) {
		this.useUnlabeledTestset = useUnlabeledTestset;
	}

	public Instances getUnlabeledTestInstances() {
		return unlabeledTestInstances;
	}

	public void setUnlabeledTestInstances(Instances unlabeledTestInstances) {
		this.unlabeledTestInstances = unlabeledTestInstances;
	}

	public void useLeskOverlap(boolean leskOverlap) {
		this.useleskOverlap = leskOverlap;

	}

	public boolean isLeskOverlapUsed() {
		return this.useleskOverlap;

	}

	public boolean isTitleQuestionMarkUsed() {
		return useTitleQuestionMark;
	}

	public void useTitleQuestionMark(boolean useTitleQuestionMark) {
		this.useTitleQuestionMark = useTitleQuestionMark;
	}

	public boolean isHypernymsSimilarityUsed() {
		return useHypernymsSimilarity;
	}

	public void useHypernymsSimilarity(boolean useHypernymsSimilarity) {
		this.useHypernymsSimilarity = useHypernymsSimilarity;
	}

	public boolean isTitleLengthUsed() {
		return useTitleLength;
	}

	public void useTitleLength(boolean useTitleLength) {
		this.useTitleLength = useTitleLength;
	}

	public boolean isBodyBoWSummCounterFeatureUsed() {
		return useBodyBoWSummCounterFeature;
	}

	public void useBodyBoWSummCounterFeature(boolean useBodyBoWSummCounterFeature) {
		this.useBodyBoWSummCounterFeature = useBodyBoWSummCounterFeature;
	}

}
