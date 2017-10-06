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
import ude.master.thesis.stance_detection.processor.Lemmatizer;
import ude.master.thesis.stance_detection.processor.NegationFeaturesGenerator2;
import ude.master.thesis.stance_detection.processor.RelatedUnrelatedFeatureGenerator;
import ude.master.thesis.stance_detection.processor.SVOFeaturesGenerator2;
import ude.master.thesis.stance_detection.processor.SimilarityFeatures;
import ude.master.thesis.stance_detection.processor.StanfordDependencyParser2;
import ude.master.thesis.stance_detection.processor.Word2VecDataGenerator2;
import ude.master.thesis.stance_detection.util.BodySummerizer2;
import ude.master.thesis.stance_detection.util.FNCConstants;
import ude.master.thesis.stance_detection.util.LeskGlossOverlaps;
import ude.master.thesis.stance_detection.util.PPDBProcessor2;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;
import ude.master.thesis.stance_detection.wordembeddings.DocToVec;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

public class FeaturesOrganiser2 {
	final static Logger logger = Logger.getLogger(FeaturesOrganiser2.class);
	// Data
	private List<List<String>> trainingStances;
	private List<List<String>> testStances;
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
	private FileHashMap<String, int[]> trainNeg;
	private FileHashMap<String, double[]> trainW2VSim;
	private FileHashMap<String, ArrayList<Integer>> trainSVO;

	private FileHashMap<String, ArrayList<Double>> trainWordOverlap;
	private FileHashMap<String, ArrayList<Integer>> trainCoOccuranceStop;
	private FileHashMap<String, ArrayList<Integer>> trainCgram;
	private FileHashMap<String, ArrayList<Integer>> trainNgram;
	private FileHashMap<String, ArrayList<Double>> trainHypSimilarity;
	private FileHashMap<String, ArrayList<Double>> trainStrMetricCosSim;
	private FileHashMap<String, ArrayList<Double>> trainLeskOverlap;

	// Test data features
	private FileHashMap<String, ArrayList<Double>> testRootDist;
	private FileHashMap<String, double[]> testPPDB;
	private FileHashMap<String, int[]> testNeg;
	private FileHashMap<String, double[]> testW2VSim;
	private FileHashMap<String, ArrayList<Integer>> testSVO;

	private FileHashMap<String, ArrayList<Double>> testWordOverlap;
	private FileHashMap<String, ArrayList<Integer>> testCoOccuranceStop;
	private FileHashMap<String, ArrayList<Integer>> testCgram;
	private FileHashMap<String, ArrayList<Integer>> testNgram;
	private FileHashMap<String, ArrayList<Double>> testHypSimilarity;
	private FileHashMap<String, ArrayList<Double>> testStrMetricCosSim;
	private FileHashMap<String, ArrayList<Double>> testLeskOverlap;

	private FileHashMap<String, String> titlesLemmas;
	private FileHashMap<Integer, Map<Integer, String>> bodiesLemmas;

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
			trainRootDist = StanfordDependencyParser2
					.loadRootDistFeaturesAsHashFile(ProjectPaths.ROOT_DIST_FEATURE_TRAIN2);
			testRootDist = StanfordDependencyParser2
					.loadRootDistFeaturesAsHashFile(ProjectPaths.ROOT_DIST_FEATURE_TEST2);

		}

		if (useSVOFeature) {
			trainSVO = SVOFeaturesGenerator2.loadSVOFeaturesAsHashFile(ProjectPaths.SUMMED_SVO_FEATURE_TRAIN2);
			testSVO = SVOFeaturesGenerator2.loadSVOFeaturesAsHashFile(ProjectPaths.SUMMED_SVO_FEATURE_TEST2);
		}

		if (usePPDBFeature) {
			trainPPDB = PPDBProcessor2.loadPPDBFeaturesAsHashFiles(ProjectPaths.PPDB_HUNG_FEATURE_TRAIN2);
			testPPDB = PPDBProcessor2.loadPPDBFeaturesAsHashFiles(ProjectPaths.PPDB_HUNG_FEATURE_TEST2);

		}

		if (useNegFeature) {
			trainNeg = NegationFeaturesGenerator2.loadNegFeaturesAsHashFile(ProjectPaths.NEG_FEATURE_TRAIN2);
			testNeg = NegationFeaturesGenerator2.loadNegFeaturesAsHashFile(ProjectPaths.NEG_FEATURE_TEST2);
		}

		if (useWord2VecAddSimilarity) {
			trainW2VSim = Word2VecDataGenerator2.loadWord2VecFeaturesAsHashFile(ProjectPaths.W2V_SIM_ADD_TRAIN2);
			testW2VSim = Word2VecDataGenerator2.loadWord2VecFeaturesAsHashFile(ProjectPaths.W2V_SIM_ADD_TEST2);

		}

		if (useOverlapFeature) {
			trainWordOverlap = RelatedUnrelatedFeatureGenerator
					.loadWordsOverlapsFeaturesAsHashFile(ProjectPaths.TRAIN_WORD_OVERLAPS_PATH2);
			testWordOverlap = RelatedUnrelatedFeatureGenerator
					.loadWordsOverlapsFeaturesAsHashFile(ProjectPaths.TEST_WORD_OVERLAPS_PATH2);
		}

		if (useBinaryCooccurraneStopFeatures) {
			trainCoOccuranceStop = RelatedUnrelatedFeatureGenerator
					.loadCoOccStopFeaturesAsHashFile(ProjectPaths.TRAIN_COOCC_PATH2);
			testCoOccuranceStop = RelatedUnrelatedFeatureGenerator
					.loadCoOccStopFeaturesAsHashFile(ProjectPaths.TEST_COOCC_PATH2);
		}

		if (useCharGramsFeatures) {
			trainCgram = RelatedUnrelatedFeatureGenerator
					.loadCharGramsFeaturesAsHashFile(ProjectPaths.TRAIN_CGRAMS_PATH2);
			testCgram = RelatedUnrelatedFeatureGenerator.loadCharGramsFeaturesAsHashFile(ProjectPaths.TEST_CGRAMS_PATH2);
		}

		if (useWordGramsFeatures) {
			trainNgram = RelatedUnrelatedFeatureGenerator.loadNGramsFeaturesAsHashFile(ProjectPaths.TRAIN_NGRAMS_PATH2);
			testNgram = RelatedUnrelatedFeatureGenerator.loadNGramsFeaturesAsHashFile(ProjectPaths.TEST_NGRAMS_PATH2);
		}

		if (useMetricCosineSimilarity) {
			trainStrMetricCosSim = SimilarityFeatures
					.loadCosSimFeaturesAsHashFile(ProjectPaths.TRAIN_COS_SIM_STRMET_PATH2);
			testStrMetricCosSim = SimilarityFeatures
					.loadCosSimFeaturesAsHashFile(ProjectPaths.TEST_COS_SIM_STRMET_PATH2);
		}

		if (useHypernymsSimilarity) {
			trainHypSimilarity = HypernymSimilarity.loadHypSimAsHashFile(ProjectPaths.TRAIN_HYP_SIM_PATH2);
			testHypSimilarity = HypernymSimilarity.loadHypSimAsHashFile(ProjectPaths.TEST_HYP_SIM_PATH2);
		}

		if (useleskOverlap) {
			/*
			 * The old way IDictionary dict = LeskGlossOverlaps.getDictionary();
			 * lgo = new LeskGlossOverlaps(dict); lgo.useStopList(true);
			 * lgo.useLemmatiser(true);
			 */
			trainLeskOverlap = SimilarityFeatures.loadLeskFeaturesAsHashFile(ProjectPaths.TRAIN_LESK_PATH2);
			testLeskOverlap = SimilarityFeatures.loadLeskFeaturesAsHashFile(ProjectPaths.TEST_LESK_PATH2);
		}

		// Load Lemmatized data
		titlesLemmas = Lemmatizer.loadTitlesLemmasAsHashFiles(ProjectPaths.TITLES_LEMMAS);
		bodiesLemmas = Lemmatizer.loadBodiesLemmasAsHashFiles(ProjectPaths.BODIES_LEMMAS);

		trainingInstances = initializeInstances("fnc-1", trainingStances, trainingSummIdBoyMap, "");
		testInstances = initializeInstances("fnc-1", testStances, testSummIdBoyMap, "");

		if (useUnlabeledTestset)
			unlabeledTestInstances = initializeInstances("fnc-1", testStances, testSummIdBoyMap, "?");

		if (saveARFF) {
			StanceDetectionDataReader.saveInstancesToArff(arffFilename + "_train", trainingInstances);
			StanceDetectionDataReader.saveInstancesToArff(arffFilename + "_test", testInstances);

			if (useUnlabeledTestset)
				StanceDetectionDataReader.saveInstancesToArff(arffFilename + "_unlabeled_test", unlabeledTestInstances);
		}
	}

	private Instances initializeInstances(String string, List<List<String>> stances,
			HashMap<Integer, Map<Integer, String>> summIdBoyMap, String unlabeled) throws IOException {
		ArrayList<Attribute> features = new ArrayList<>();

		if (useBodyBoWCounterFeature) {
			features.add(new Attribute(FNCConstants.BODY_BOW_COUNTER, (List<String>) null));
		}
		if (useRootDistFeature) {
			for (int i = 0; i < 8; i++)
				features.add(new Attribute(FNCConstants.ROOT_DIST_REFUTE + i));

			for (int i = 0; i < 8; i++)
				features.add(new Attribute(FNCConstants.ROOT_DIST_DISCUSS + i));
		}

		if (usePPDBFeature) {
			for (int i = 0; i < 10; i++)
				features.add(new Attribute(FNCConstants.PPDB_HUNG + i));
		}

		if (useNegFeature) {
			for (int i = 0; i < 8; i++)
				features.add(new Attribute(FNCConstants.NEG_FEATURE + i));
		}

		if (useWord2VecAddSimilarity) {
			for (int i = 0; i < 3; i++)
				features.add(new Attribute(FNCConstants.WORD2VEC_ADD_SIM + i));
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
			// features.add(new Attribute(FNCConstants.WORD_OVERLAP));
			for (int i = 0; i < 9; i++) {
				features.add(new Attribute(FNCConstants.WORD_OVERLAP + i));
			}
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
			for (int i = 0; i < 9; i++) {
				features.add(new Attribute(FNCConstants.BINARY_COOCCURANCE_STOP_COUNT + i));
			}
		}

		if (useCharGramsFeatures) {
			int featSize = (3 * BodySummerizer2.NUM_SENT_BEG) + (3 * BodySummerizer2.NUM_SENT_END)
					+ 3 * 3;
			for (int i = 0; i < featSize; i++) {
				features.add(new Attribute(FNCConstants.CHAR_GRAMS_HITS + i));
			}
		}

		if (useWordGramsFeatures) {			
			int featSize = (4 * BodySummerizer2.NUM_SENT_BEG) + (4 * BodySummerizer2.NUM_SENT_END)
					+ 4 * 2;
			for (int i = 0; i < featSize; i++) {
				features.add(new Attribute(FNCConstants.NGRAM_HITS + i));
			}
		}

		if (useMetricCosineSimilarity) {
			// features.add(new Attribute(FNCConstants.COSINE_METRIC_SIM));
			for (int i = 0; i < 9; i++) {
				features.add(new Attribute(FNCConstants.COSINE_METRIC_SIM + i));
			}
		}

		if (useleskOverlap) {
			features.add(new Attribute(FNCConstants.LESK_OVERLAP));
		}

		if (useHypernymsSimilarity) {
			for (int i = 0; i < 9; i++) {
				features.add(new Attribute(FNCConstants.HYP_SIM + i));
			}
		}

		if (useTitleQuestionMark) {
			features.add(new Attribute(FNCConstants.TITLE_Q));
		}
		
		if(useTitleLength){
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

		assignInstancesValues(stances, summIdBoyMap, instances, unlabeled, features.size());
		return instances;
	}

	private void assignInstancesValues(List<List<String>> stances, HashMap<Integer, Map<Integer, String>> summIdBoyMap,
			Instances instances, String unlabeled, int featuresSize) throws IOException {
		System.out.println("Started getting instances");
		int i = 0;
		for (List<String> stance : stances) {
			// this if stmt is related to how many classes are used in testing
			// the classifier
			if (useBinaryRelatedUnrelatedClasses) {

				String headline = stance.get(0);
				// String bodyPart1 = FeatureExtractor
				// .getLemmatizedCleanStr(summIdBoyMap.get(Integer.valueOf(stance.get(1))).get(1));
				// String bodyPart2 = FeatureExtractor
				// .getLemmatizedCleanStr(summIdBoyMap.get(Integer.valueOf(stance.get(1))).get(3));
				String bodyPart1 = (bodiesLemmas.get(Integer.valueOf(stance.get(1))).get(1));
				String bodyPart2 = (bodiesLemmas.get(Integer.valueOf(stance.get(1))).get(3));

				ArrayList<String> bodyParts = new ArrayList<>();
				bodyParts.add(bodyPart1);
				bodyParts.add(bodyPart2);

				DenseInstance instance = createInstance(headline, bodyParts, stance.get(1), instances, featuresSize);
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

					String bodyPart1 = FeatureExtractor
							.getLemmatizedCleanStr(summIdBoyMap.get(Integer.valueOf(stance.get(1))).get(1));
					String bodyPart2 = FeatureExtractor
							.getLemmatizedCleanStr(summIdBoyMap.get(Integer.valueOf(stance.get(1))).get(3));

					ArrayList<String> bodyParts = new ArrayList<>();
					bodyParts.add(bodyPart1);
					bodyParts.add(bodyPart2);

					DenseInstance instance = createInstance(headline, bodyParts, stance.get(1), instances,
							featuresSize);
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
	 * @param bodyParts
	 * @param bodyId
	 * @param instances
	 * @param featuresSize
	 * @param instanceType
	 * @return
	 * @throws IOException
	 */
	private DenseInstance createInstance(String headline, ArrayList<String> bodyParts, String bodyId,
			Instances instances, int featuresSize) throws IOException {
		DenseInstance instance = new DenseInstance(featuresSize);
		instance.setDataset(instances);

		if (useBodyBoWCounterFeature) {
			instance.setValue(instances.attribute(FNCConstants.BODY_BOW_COUNTER),
					bodyParts.get(0) + " " + bodyParts.get(1));
		}

		if (useRootDistFeature) {
			ArrayList<Double> rootdist = null;

			rootdist = trainRootDist.get(bodyId);
			if (rootdist == null)
				rootdist = testRootDist.get(bodyId);

			for (int i = 0; i < rootdist.size()/2; i++)
				instance.setValue(instances.attribute(FNCConstants.ROOT_DIST_REFUTE + i), rootdist.get(i));
			for (int i = 0; i < rootdist.size()/2; i++)
				instance.setValue(instances.attribute(FNCConstants.ROOT_DIST_DISCUSS + i), rootdist.get(i + rootdist.size()/2));

		}

		if (usePPDBFeature) {
			double[] ppdb = null;

			ppdb = trainPPDB.get(headline + bodyId);
			if (ppdb == null)
				ppdb = testPPDB.get(headline + bodyId);

			for (int i = 0; i < 10; i++) {
				instance.setValue(instances.attribute(FNCConstants.PPDB_HUNG + i), ppdb[i]);
			}

		}

		if (useNegFeature) {
			int[] neg = null;

			neg = trainNeg.get(headline + bodyId);
			if (neg == null)
				neg = testNeg.get(headline + bodyId);

			for(int i = 0; i < neg.length; i++)
				instance.setValue(instances.attribute(FNCConstants.NEG_FEATURE + i), neg[i]);
		}

		if (useWord2VecAddSimilarity) {
			double[] sim = null;

			sim = trainW2VSim.get(headline + bodyId);
			if (sim == null)
				sim = testW2VSim.get(headline + bodyId);
			
			for(int i = 0; i < sim.length; i++)
				instance.setValue(instances.attribute(FNCConstants.WORD2VEC_ADD_SIM + i), sim[i]);
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
			ArrayList<Double> wordOverlap = null;
			wordOverlap = trainWordOverlap.get(headline + bodyId);
			if (wordOverlap == null)
				wordOverlap = testWordOverlap.get(headline + bodyId);

			for (int i = 0; i < 9; i++) {
				Attribute wordOverlapAtt = instances.attribute(FNCConstants.WORD_OVERLAP + i);
				instance.setValue(wordOverlapAtt, wordOverlap.get(i));
			}
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
			instance.setValue(bodyPolarityAtt,
					FeatureExtractorWithModifiedBL.calculatePolarity(bodyParts.get(0) + " " + bodyParts.get(1)));
		}
		// TODO: split to 2 features use
		if (useBinaryCooccurraneFeatures) {
			List<Integer> binCoOccFeats = FeatureExtractorWithModifiedBL.getBinaryCoOccurenceFeatures(headline,
					bodyParts.get(0) + " " + bodyParts.get(1));
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

			for (int i = 0; i < 9; i++) {
				Attribute coOccuranceStopAtt = instances.attribute(FNCConstants.BINARY_COOCCURANCE_STOP_COUNT + i);
				instance.setValue(coOccuranceStopAtt, coOccuranceStop.get(i));
			}
		}

		if (useCharGramsFeatures) {
			ArrayList<Integer> cgram = null;
			cgram = trainCgram.get(headline + bodyId);
			if (cgram == null)
				cgram = testCgram.get(headline + bodyId);
			
			int featSize = (3 * BodySummerizer2.NUM_SENT_BEG) + (3 * BodySummerizer2.NUM_SENT_END)
					+ 3 * 3;

			for (int i = 0; i < featSize; i++) {
				Attribute cgramAtt = instances.attribute(FNCConstants.CHAR_GRAMS_HITS + i);
				instance.setValue(cgramAtt, cgram.get(i));
			}
		}

		if (useWordGramsFeatures) {
			int featSize = (4 * BodySummerizer2.NUM_SENT_BEG) + (4 * BodySummerizer2.NUM_SENT_END)
					+ 4 * 2;
			
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
			ArrayList<Double> lesk = null;
			lesk = trainLeskOverlap.get(headline + bodyId);
			if (lesk == null)
				lesk = testLeskOverlap.get(headline + bodyId);

			for (int i = 0; i < 1; i++) {
				Attribute ngramAtt = instances.attribute(FNCConstants.LESK_OVERLAP);
				instance.setValue(ngramAtt, lesk.get(i));
			}
		}

		if (useTitleQuestionMark) {
			int q = 0;
			if (headline.contains("?"))
				q = 1;
			Attribute qAtt = instances.attribute(FNCConstants.TITLE_Q);
			instance.setValue(qAtt, q);
		}
		
		if(useTitleLength){
			int len = headline.split("\\s+").length;
			Attribute lenAtt = instances.attribute(FNCConstants.TITLE_LENGTH);
			instance.setValue(lenAtt, len);
		}

		if (useMetricCosineSimilarity) {
			ArrayList<Double> cos = null;
			cos = trainStrMetricCosSim.get(headline + bodyId);
			if (cos == null)
				cos = testStrMetricCosSim.get(headline + bodyId);

			for (int i = 0; i < 9; i++) {
				Attribute cosAtt = instances.attribute(FNCConstants.COSINE_METRIC_SIM + i);
				instance.setValue(cosAtt, cos.get(i));
			}
		}

		if (useHypernymsSimilarity) {
			ArrayList<Double> hyp = null;
			hyp = trainHypSimilarity.get(headline + bodyId);
			if (hyp == null)
				hyp = testHypSimilarity.get(headline + bodyId);

			for (int i = 0; i < 9; i++) {
				Attribute cosAtt = instances.attribute(FNCConstants.HYP_SIM + i);
				instance.setValue(cosAtt, hyp.get(i));
			}
		}

		return instance;
	}

	public void loadData() throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_LESS2_PREPROCESSED, ProjectPaths.SUMMARIZED_TRAIN_BODIES2,
				ProjectPaths.TEST_STANCESS_LESS2_PREPROCESSED, ProjectPaths.SUMMARIZED_TEST_BODIES2);

		trainingSummIdBoyMap = sddr.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TRAIN_BODIES));
		testSummIdBoyMap = sddr.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TEST_BODIES));

		trainingStances = sddr.getTrainStances();
		System.out.println(trainingStances.size());

		testStances = sddr.getTestStances();
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
		trainingInstances = trainingInstances;
	}

	public Instances getTestInstances() {
		return testInstances;
	}

	public void setTestInstances(Instances testInstances) {
		testInstances = testInstances;
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

}
