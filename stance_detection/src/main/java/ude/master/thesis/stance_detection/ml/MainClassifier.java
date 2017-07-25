package ude.master.thesis.stance_detection.ml;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ude.master.thesis.stance_detection.processor.FeatureExtractor;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

public class MainClassifier {

	// Baseline features settings
	// TODO: split the feature types in a better way (like in hs project)
	private boolean useOverlapFeature;
	private boolean useRefutingFeatures;
	private boolean usePolarityFeatures;
	private boolean useBinaryCooccurraneFeatures;
	private boolean useBinaryCooccurraneStopFeatures; // Same as binary
														// cooccurrence but
														// ignores stopwords.
	private boolean useCharGramsFeatures;
	private boolean useWordGramsFeatures;

	private Instances initializeInstances(String relationName, List<List<String>> stances,
			Map<Integer, String> idBodyMap) {
		ArrayList<Attribute> features = new ArrayList<>();

		/**
		 * TODO: When using BOW filter: 1. we add the text value as is as a
		 * seprate attribute 2. apply the StringToWord vector on those attribute
		 * 3. see initializeBOWFilter() in hs and useMessage in
		 * initializeInstances(...)
		 */
		if (useOverlapFeature) {
			features.add(new Attribute("word_overlap"));
		}

		if (useRefutingFeatures) {
			for (String refute : FeatureExtractor.refutingWords) {
				features.add(new Attribute("refute_" + refute));
			}
		}

		if (usePolarityFeatures) {
			features.add(new Attribute("pol_head"));
			features.add(new Attribute("pol_body"));
		}

		if (useBinaryCooccurraneFeatures) {
			features.add(new Attribute("bin_co_occ_count"));
			features.add(new Attribute("bin_co_occ_255")); // just in the first
															// 255 words
		}

		if (useBinaryCooccurraneStopFeatures) {
			features.add(new Attribute("bin_co_occ_stop_count"));
			features.add(new Attribute("bin_co_occ_stop_255"));
		}

		if (useCharGramsFeatures) {
			int[] cgramSizes = { 2, 8, 4, 16 };
			for (int size : cgramSizes) {
				features.add(new Attribute("cgram_hits_" + size));
				features.add(new Attribute("cgram_early_hits_" + size));
				features.add(new Attribute("cgram_first_hits_" + size));
				// TODO: Not in baseline
				// features.add(new Attribute("cgram_tail_hits_" + size));
			}
		}

		if (useWordGramsFeatures) {
			int[] ngramSizes = { 2, 3, 4, 5, 6 };
			for (int size : ngramSizes) {
				features.add(new Attribute("ngram_hits_" + size));
				features.add(new Attribute("ngram_early_hits_" + size));
				// TODO: Not in baseline
				// features.add(new Attribute("ngram_tail_hits" + size));
			}
		}

		Instances instances = new Instances("fnc-1", features, stances.size());

		instances.setClassIndex(features.size() - 1);

		assignFeaturesValues(stances, idBodyMap, instances, features.size());
		return instances;
	}

	private void assignFeaturesValues(List<List<String>> stances, Map<Integer, String> idBodyMap, Instances instances,
			int featuresSize) {

		for (List<String> stance : stances) {
			String headline = stance.get(0);
			String body = idBodyMap.get(Integer.valueOf(stance.get(1)));

			DenseInstance instance = createInstance(headline, body, instances, featuresSize);
		}

	}

	/**
	 * Converts the title and body pair to features and wrap them in an Instance
	 * object
	 * 
	 * @param headline
	 * @param body
	 * @param instances
	 * @param featuresSize
	 * @return
	 */
	private DenseInstance createInstance(String headline, String body, Instances instances, int featuresSize) {
		// Create instance and set the number of features
		DenseInstance instance = new DenseInstance(featuresSize);

		instance.setDataset(instances);

		if (useOverlapFeature) {
			Attribute wordOverlapAtt = instances.attribute("word_overlap");
			instance.setValue(wordOverlapAtt, FeatureExtractor.getWordOverlapFeature(headline, body));
		}

		if (useRefutingFeatures) {
			for (String refute : FeatureExtractor.refutingWords) {
				Attribute refuteAtts = instances.attribute("refute_" + refute);
				instance.setValue(refuteAtts, FeatureExtractor.getRefutingFeature(headline, refute));
			}
		}

		// TODO: split to 2 features use
		if (usePolarityFeatures) {
			for (String refute : FeatureExtractor.refutingWords) {
				Attribute headPolarityAtt = instances.attribute("pol_head");
				instance.setValue(headPolarityAtt, FeatureExtractor.getPolarityFeatures(headline, refute));

				Attribute bodyPolarityAtt = instances.attribute("pol_body");
				instance.setValue(bodyPolarityAtt, FeatureExtractor.getPolarityFeatures(body, refute));
			}
		}
		// TODO: split to 2 features use
		if (useBinaryCooccurraneFeatures) {
			Attribute binCoOccAtt = instances.attribute("bin_co_occ_count");
			instance.setValue(binCoOccAtt, FeatureExtractor.getBinaryCoOccurenceFeatures(headline, body).get(0));

			Attribute binCoOccEarlyAtt = instances.attribute("bin_co_occ_255");
			instance.setValue(binCoOccEarlyAtt, FeatureExtractor.getBinaryCoOccurenceFeatures(headline, body).get(1));
		}

		// TODO: split to 2 features use
		if (useBinaryCooccurraneStopFeatures) {
			List<Integer> f = FeatureExtractor.getBinaryCoOccurenceStopFeatures(headline, body);
			Attribute binCoOccAtt = instances.attribute("bin_co_occ_stop_count");
			instance.setValue(binCoOccAtt, f.get(0));

			Attribute binCoOccEarlyAtt = instances.attribute("bin_co_occ_stop_255");
			instance.setValue(binCoOccEarlyAtt, f.get(1));
		}

		// TODO: split to 3 features use
		if (useCharGramsFeatures) {
			int[] cgramSizes = { 2, 8, 4, 16 };
			for (int size : cgramSizes) {
				List<Integer> f = FeatureExtractor.getCharGramsFeatures(headline, body, size);

				Attribute cgramHitsAtt = instances.attribute("cgram_hits_" + size);
				instance.setValue(cgramHitsAtt, f.get(0));

				Attribute cgramEarlyHitsAtt = instances.attribute("cgram_early_hits_" + size);
				instance.setValue(cgramEarlyHitsAtt, f.get(1));

				Attribute cgramFirstHitsAtt = instances.attribute("cgram_first_hits_" + size);
				instance.setValue(cgramFirstHitsAtt, f.get(2));
				// TODO: Not in baseline
				// features.add(new Attribute("cgram_tail_hits_" + size));
			}

		}

		if (useWordGramsFeatures) {
			int[] ngramSizes = { 2, 3, 4, 5, 6 };
			for (int size : ngramSizes) {
				List<Integer> f = FeatureExtractor.getNGramsFeatures(headline, body, size);
				
				Attribute ngramHitAtt = instances.attribute("ngram_hits_" + size);
				instance.setValue(ngramHitAtt, f.get(0));
				
				Attribute ngramEarlyHitsAtt = instances.attribute("ngram_early_hits_" + size);
				instance.setValue(ngramEarlyHitsAtt, f.get(1));
			}
		}

		return instance;
	}

	public boolean isUseOverlapFeature() {
		return useOverlapFeature;
	}

	public void setUseOverlapFeature(boolean useOverlapFeature) {
		this.useOverlapFeature = useOverlapFeature;
	}

	public boolean isUseRefutingFeatures() {
		return useRefutingFeatures;
	}

	public void setUseRefutingFeatures(boolean useRefutingFeatures) {
		this.useRefutingFeatures = useRefutingFeatures;
	}

	public boolean isUsePolarityFeatures() {
		return usePolarityFeatures;
	}

	public void setUsePolarityFeatures(boolean usePolarityFeatures) {
		this.usePolarityFeatures = usePolarityFeatures;
	}

	public boolean isUseBinaryCooccurraneFeatures() {
		return useBinaryCooccurraneFeatures;
	}

	public void setUseBinaryCooccurraneFeatures(boolean useBinaryCooccurraneFeatures) {
		this.useBinaryCooccurraneFeatures = useBinaryCooccurraneFeatures;
	}

	public boolean isUseCharGramsFeatures() {
		return useCharGramsFeatures;
	}

	public void setUseCharGramsFeatures(boolean useCharGramsFeatures) {
		this.useCharGramsFeatures = useCharGramsFeatures;
	}

	public boolean isUseWordGramsFeatures() {
		return useWordGramsFeatures;
	}

	public void setUseWordGramsFeatures(boolean useWordGramsFeatures) {
		this.useWordGramsFeatures = useWordGramsFeatures;
	}

	public boolean isUseBinaryCooccurraneStopFeatures() {
		return useBinaryCooccurraneStopFeatures;
	}

	public void setUseBinaryCooccurraneStopFeatures(boolean useBinaryCooccurraneStopFeatures) {
		this.useBinaryCooccurraneStopFeatures = useBinaryCooccurraneStopFeatures;
	}

}
