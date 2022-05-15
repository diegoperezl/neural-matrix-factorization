package stacking;

import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.User;
import es.upm.etsisi.cf4j.recommender.Recommender;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.BNMF;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.BiasedMF;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.NMF;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.PMF;
import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class Stacking5 extends Recommender {

    /** Neural network * */
    private final ComputationGraph network;

    /** Number of epochs * */
    private final int numEpochs;

    /** Learning Rate */
    protected final double learningRate;

    /** Map of Recommenders */
    protected final Map<String, Recommender> recommenders;

    /**
     * Model constructor from a Map containing the model's hyper-parameters values. Map object must
     * contains the following keys:
     *
     * <ul>
     *   <li><b>numEpochs</b>: int value with the number of epochs.
     *   <li><b>learningRate</b>: double value with the learning rate.
     *   <li><b><em>seed</em></b> (optional): random seed for random numbers generation. If missing,
     *       random value is used.
     * </ul>
     *
     * @param datamodel DataModel instance
     * @param params Model's hyper-parameters values
     */
    public Stacking5(DataModel datamodel, Map<String, Object> params) {
        this(
                datamodel,
                (int) params.get("numEpochs"),
                (double) params.get("learningRate"),
                params.containsKey("seed") ? (long) params.get("seed") : System.currentTimeMillis());
    }

    /**
     * Model constructor
     *
     * @param datamodel DataModel instance
     * @param numEpochs Number of epochs
     * @param learningRate Learning rate
     */
    public Stacking5(
            DataModel datamodel,
            int numEpochs,
            double learningRate) {
        this(
                datamodel,
                numEpochs,
                learningRate,
                System.currentTimeMillis());
    }

    /**
     * Model constructor
     *
     * @param datamodel DataModel instance
     * @param numEpochs Number of epochs
     * @param learningRate Learning rate
     * @param seed Seed for random numbers generation
     */
    public Stacking5(
            DataModel datamodel,
            int numEpochs,
            double learningRate,
            long seed) {
        super(datamodel);

        this.numEpochs = numEpochs;
        this.learningRate = learningRate;

        this.recommenders = new HashMap<>();

        this.recommenders.put("BiasedMF", new BiasedMF(datamodel, 4, 100, 0.1, 0.01, 42L));
        this.recommenders.put("PMF", new PMF(datamodel, 4, 100, 0.1, 0.005, 42L));
        this.recommenders.put("NMF", new NMF(datamodel, 8, 100, 42L));
        this.recommenders.put("BNMF", new BNMF(datamodel, 8, 100, 0.8, 10.0, 42L));

        this.recommenders.get("BiasedMF").fit();
        this.recommenders.get("PMF").fit();
        this.recommenders.get("NMF").fit();
        this.recommenders.get("BNMF").fit();

        String recommendersName[] = new String[this.recommenders.size()];
        int index = 0;
        for (String recommenderName : recommenders.keySet()) {
            recommendersName[index] = recommenderName;
            index++;
        }

        String[] layersName = ArrayUtils.addAll(new String[] {"user", "item"}, recommendersName);

        ComputationGraphConfiguration.GraphBuilder builder =
                new NeuralNetConfiguration.Builder().seed(seed).graphBuilder().addInputs(layersName);

        String[] concatLayers = new String[recommenders.size()];
        index = 0;
        for (String recommender : recommenders.keySet()) {
            builder
                    .addLayer(
                            recommender + "Layer",
                            new DenseLayer.Builder()
                                    .nIn(datamodel.getNumberOfUsers() + datamodel.getNumberOfItems() + 1)
                                    .nOut(1)
                                    .updater(new Adam(learningRate))
                                    .activation(Activation.IDENTITY)
                                    .build(),
                            "user", "item", recommender);

            concatLayers[index] = recommender + "Layer";
            index++;
        }

        builder
                .addLayer(
                        "out",
                        new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                                .nIn(recommenders.size())
                                .nOut(1)
                                .weightInit(WeightInit.ONES)
                                .updater(new Sgd(0.0))
                                //.updater(new Adam(learningRate))
                                .activation(Activation.IDENTITY)
                                .build(),
                        concatLayers)
                .setOutputs("out")
                .build();

        this.network = new ComputationGraph(builder.build());
        this.network.init();
    }

    @Override
    public void fit() {
        System.out.println("\nFitting " + this.toString());

        int numRecommenders = this.recommenders.size();

        NDArray[] X = new NDArray[2 + numRecommenders];
        NDArray[] y = new NDArray[1];

        double[][] users =
                new double[super.getDataModel().getNumberOfRatings()][datamodel.getNumberOfUsers()];
        double[][] items =
                new double[super.getDataModel().getNumberOfRatings()][datamodel.getNumberOfItems()];
        double[][] ratings = new double[super.getDataModel().getNumberOfRatings()][1];
        double[][][] recommendersPredictions =
                new double[numRecommenders][super.getDataModel().getNumberOfRatings()][1];

        int i = 0;
        for (User user : super.datamodel.getUsers()) {
            for (int pos = 0; pos < user.getNumberOfRatings(); pos++) {
                int itemIndex = user.getItemAt(pos);

                users[i][user.getUserIndex()] = 1;
                items[i][itemIndex] = 1;
                ratings[i][0] = user.getRatingAt(pos);

                int indeX = 0;
                for (Map.Entry<String, Recommender> recommender : recommenders.entrySet()) {
                    recommendersPredictions[indeX][i][0] =
                            recommender.getValue().predict(user.getUserIndex(), itemIndex);
                    indeX++;
                }

                i++;
            }
        }

        X[0] = new NDArray(users);
        X[1] = new NDArray(items);
        y[0] = new NDArray(ratings);

        int indeX = 2;
        for (Map.Entry<String, Recommender> recommender : recommenders.entrySet()) {
            X[indeX] = new NDArray(recommendersPredictions[indeX - 2]);
            indeX++;
        }

        for (int epoch = 1; epoch <= this.numEpochs; epoch++) {
            this.network.fit(X, y);
            if ((epoch % 5) == 0) System.out.print(".");
            if ((epoch % 50) == 0) System.out.println(epoch + " iterations");
        }
    }

    /**
     * Returns the prediction of a rating of a certain user for a certain item, through these
     * predictions the metrics of MAE, MSE and RMSE can be obtained.
     *
     * @param userIndex Index of the user in the array of Users of the DataModel instance
     * @param itemIndex Index of the item in the array of Items of the DataModel instance
     * @return Prediction
     */
    public double predict(int userIndex, int itemIndex) {
        int numRecommenders = this.recommenders.size();

        NDArray[] X = new NDArray[2 + numRecommenders];

        double[][] users = new double[1][datamodel.getNumberOfUsers()];
        double[][] items = new double[1][datamodel.getNumberOfItems()];

        users[0][userIndex] = 1;
        items[0][itemIndex] = 1;

        X[0] = new NDArray(users);
        X[1] = new NDArray(items);

        int indeX = 2;
        for (Map.Entry<String, Recommender> recommender : recommenders.entrySet()) {
            X[indeX] =
                    new NDArray(new double[][] {{recommender.getValue().predict(userIndex, itemIndex)}});
            indeX++;
        }

        INDArray[] y = this.network.output(X);

        return y[0].toDoubleVector()[0];
    }

    /**
     * Returns the number of epochs.
     *
     * @return Number of epochs.
     */
    public int getNumEpochs() {
        return this.numEpochs;
    }

    /**
     * Returns learning rate.
     *
     * @return learning rate.
     */
    public double getLearningRate() {
        return this.learningRate;
    }

    @Override
    public String toString() {
        StringBuilder str =
                new StringBuilder("StackingMF(")
                        .append("numEpochs=")
                        .append(this.numEpochs)
                        .append("; ")
                        .append("learningRate=")
                        .append(this.learningRate)
                        .append(")");
        return str.toString();
    }
}
