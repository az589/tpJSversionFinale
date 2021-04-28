// Note: Require the cpu and webgl backend and add them to package.json as peer dependencies.
require('@tensorflow/tfjs-node-gpu');
const cocoSsd = require('@tensorflow-models/coco-ssd');
const fs = require('fs-extra');
const jpeg = require('jpeg-js');
const Bromise = require('bluebird');
const R = require('ramda');

const imagesFolder = './images/';
const addFolderPath = img => './images/' + img;
const addPathToList = (list, path) => R.append(path, list);

const getImagesPath = folder => {
    let listPaths = [];
    fs.readdirSync(folder).forEach(file => {
        let fullPath = addFolderPath(file);
        listPaths = addPathToList(listPaths, fullPath);
    });
    return listPaths;
};

const imagesFullPaths = getImagesPath(imagesFolder);
console.log('Récupération des chemins des images dans le dossier ./images :');
console.log(imagesFullPaths);

const readJpg = async (path) => jpeg.decode(await fs.readFile(path), true);

(async () => {
    const imgList = await Bromise.map(
        imagesFullPaths,
        readJpg
    );

    // Load the model.
    const model = await cocoSsd.load();

    // Classify the image.
    const predictions = await Bromise.map(imgList, (x) => model.detect(x));

    let pathIdx = 0;

    const addPath = (list, obj) => {
        obj = R.assoc('path', list[pathIdx], obj);
        pathIdx = R.inc(pathIdx);
        return obj;
    };

    const addPathDeleteScoreAndBBox = list => R.map(
        R.pipe((item) => addPath(imagesFullPaths, item),
            R.dissoc('score'),
            R.dissoc('bbox')
        ), list);

    const predictionsClassAndPath = addPathDeleteScoreAndBBox(R.flatten(predictions));

    const putImageInDirectory = prediction => {
        const theClass = prediction.class;
        const thePath = prediction.path;
        const pathDirectory = './sortedImages/' + theClass;
        const pathImage = R.replace('./images/', '/', thePath);
        const goodPathImage = pathDirectory + pathImage;
        fs.mkdirSync(pathDirectory, { recursive: true });
        fs.copy(thePath, goodPathImage, function(err){
            if (err) return console.error(err);
            console.log('L\'image ' + pathImage + ' a bien été copiée dans le dossier ./sortedImages.');
            console.log('Elle a été placée dans le sous-dossier : ' + theClass);
        });
        return prediction;
    };

    const classifyImagesOneByOne = list => {
        R.map(putImageInDirectory, list);
    };

    classifyImagesOneByOne(predictionsClassAndPath);
})();