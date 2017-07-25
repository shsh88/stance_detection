package ude.master.thesis.stance_detection.ml;

public class MainClassifier {

	// TODO: These vocabulary needs stemming to be more precise, then needs to
	// be compared to words from input after stemming (e.g. no need to have
	// doubt & doubts together)
	private static String[] refutingWords = { "fake", "fraud", "hoax", "false", "deny", "denies", "not", "despite",
			"nope", "doubt", "doubts", "bogus", "debunk", "pranks", "retract" };

	// TODO: Needs investigation--> look at what words used in discussing
	// articles (Maybe find the words that shows more often / intersect)
	private static String[] discussWords = { "according", "maybe", "reporting", "reports", "say", "says", "claim",
			"claims", "purportedly", "investigating", "told", "tells", "allegedly", "validate", "verify" };

	
	//Baseline features settings
	//TODO: split the feature types in a better way (like in hs project)
	private boolean useOverlapFeature;
	private boolean useRefutingFeatures;
	private boolean usePolarityFeatures;
	private boolean useBinaryCooccurraneFeatures;
	private boolean useCharGramsFeatures;
	private boolean useWordGramsFeatures;
	
	
}
