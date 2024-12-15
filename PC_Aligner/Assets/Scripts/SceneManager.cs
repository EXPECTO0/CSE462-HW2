using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using TMPro;


public class SceneManager : MonoBehaviour
{
    [SerializeField] private string pc1FileName;
    [SerializeField] private string pc2FileName;
    [SerializeField] private float alignmentThreshold = 1f; 
    [SerializeField] private float numIterations = 200000; 
    [SerializeField] private  TextMeshProUGUI transformationText;
    public Material material1;
    public Material material2;
    public Material material3;
    public Material lineMaterial;
    private List<Vector3> pointCloud1;
    private List<Vector3> pointCloud2;
    private List<Vector3> alignedPointCloud;
    private Matrix4x4 transformationMatrix;

    private enum ViewMode { ColoredPoints, MovementLines }
    private ViewMode currentViewMode = ViewMode.ColoredPoints;
    private List<GameObject> pointObjects = new List<GameObject>();
    private List<GameObject> lineObjects = new List<GameObject>();


    void Start()
    {
        pointCloud1 = LoadPointCloud(pc1FileName);
        pointCloud2 = LoadPointCloud(pc2FileName);

        transformationMatrix = AlignPointCloudsWithRANSAC(pointCloud1, pointCloud2);
        alignedPointCloud = TransformPointCloud(pointCloud1, transformationMatrix);

        ShowColoredPoints();
        ShowTransformationAndRotation(transformationMatrix);



        // Combine all points for camera adjustment
        List<Vector3> allPoints = new List<Vector3>();
        allPoints.AddRange(pointCloud1);
        allPoints.AddRange(pointCloud2);
        allPoints.AddRange(alignedPointCloud);

        // Adjust the camera to focus on all points
        Camera mainCamera = Camera.main;
        PositionCameraForPoints(mainCamera, allPoints);


    }
    public void ShowTransformationAndRotation(Matrix4x4 transformationMatrix)
    {
        // Extract translation vector
        Vector3 translation = transformationMatrix.GetColumn(3);

        // Extract rotation matrix
        Matrix4x4 rotationMatrix = Matrix4x4.identity;
        rotationMatrix.SetColumn(0, transformationMatrix.GetColumn(0));
        rotationMatrix.SetColumn(1, transformationMatrix.GetColumn(1));
        rotationMatrix.SetColumn(2, transformationMatrix.GetColumn(2));

        // Format translation vector
        string translationString = $"Translation Matrix:\n" +
                                   $"X: {translation.x:F3}, Y: {translation.y:F3}, Z: {translation.z:F3}";

        // Format rotation matrix
        string rotationString = $"Rotation Matrix:\n" +
                                $"{rotationMatrix[0, 0]:F3}, {rotationMatrix[0, 1]:F3}, {rotationMatrix[0, 2]:F3}\n" +
                                $"{rotationMatrix[1, 0]:F3}, {rotationMatrix[1, 1]:F3}, {rotationMatrix[1, 2]:F3}\n" +
                                $"{rotationMatrix[2, 0]:F3}, {rotationMatrix[2, 1]:F3}, {rotationMatrix[2, 2]:F3}";

        // Combine into one string
        string displayText = $"{translationString}\n\n{rotationString}";

        // Display on screen
        if (transformationText != null)
        {
            transformationText.text = displayText;
        }
        else
        {
            Debug.LogError("TransformationText UI element is not assigned!");
        }
    }
    List<Vector3> LoadPointCloud(string fileName)
    {
        string filePath = Path.Combine(Application.dataPath, "Resources", $"{fileName}.txt");
        List<Vector3> points = new List<Vector3>();
        string[] lines = File.ReadAllLines(filePath);

        int numPoints = int.Parse(lines[0]);
        for (int i = 1; i <= numPoints; i++)
        {
            string[] parts = lines[i].Split(' ');
            float x = float.Parse(parts[0]);
            float y = float.Parse(parts[1]);
            float z = float.Parse(parts[2]);
            points.Add(new Vector3(x, y, z));
        }
        return points;
    }
    public void ToggleView()
    {
        ClearScene();

        if (currentViewMode == ViewMode.ColoredPoints)
        {
            currentViewMode = ViewMode.MovementLines;
            Debug.Log("ShowMovementLines");
            ShowMovementLines();
        }
        else
        {
            currentViewMode = ViewMode.ColoredPoints;
            Debug.Log("ShowColoredPoints");
            ShowColoredPoints();
        }
    }
    void ShowColoredPoints()
    {
        ShowPointCloud(pointCloud1, material1, 'P');
        ShowPointCloud(pointCloud2, material2, 'Q');
        ShowPointCloud(alignedPointCloud, material3, 'M');
    }

    void ShowPointCloud(List<Vector3> pointCloud, Material material, char objectTagChar)
    {
        foreach (var point in pointCloud)
        {
            GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere.transform.position = point;
            sphere.GetComponent<Renderer>().material = material;
            sphere.name = $"{objectTagChar}_Sphere_{pointCloud.IndexOf(point)}";
            sphere.transform.parent = transform;
            pointObjects.Add(sphere);
        }
    }
    void ShowMovementLines()
    {
        ShowPointCloud(pointCloud1, material1, 'P');
        ShowPointCloud(alignedPointCloud, material3, 'M');

        for (int i = 0; i < alignedPointCloud.Count; i++)
        {
            // Draw line between the original and aligned points
            GameObject line = new GameObject($"Line_{i}");
            LineRenderer lr = line.AddComponent<LineRenderer>();
            lr.startWidth = 0.3f;
            lr.endWidth = 0.3f;
            lr.material = lineMaterial;
            lr.positionCount = 2;
            lr.SetPositions(new Vector3[] { pointCloud1[i], alignedPointCloud[i] });

            line.transform.parent = transform; // Assign to SceneManage
            lineObjects.Add(line);
        }
    }
    void ClearScene()
    {
        foreach (var obj in pointObjects) 
            Destroy(obj);
        foreach (var obj in lineObjects) 
            Destroy(obj);
        
        pointObjects.Clear();
        lineObjects.Clear();
    }
    private Matrix4x4 AlignPointCloudsWithRANSAC(List<Vector3> pc1, List<Vector3> pc2)
    {
        int maxInliers = 0;
        Matrix4x4 bestTransformation = Matrix4x4.identity;

        for (int i = 0; i < numIterations; i++)
        {
            var sampledPc1 = SampleRandomPoints(pc1, 3);
            var sampledPc2 = SampleRandomPoints(pc2, 3);

            Matrix4x4 transformation = ComputeRigidTransformation(sampledPc1, sampledPc2);

            if (transformation != Matrix4x4.identity)
            {
                int inliers = CountInliers(pc1, pc2, transformation, alignmentThreshold);

                if (inliers > maxInliers)
                {
                    maxInliers = inliers;
                    bestTransformation = transformation;
                }
            }
        }

        return bestTransformation;
    }

    private List<Vector3> SampleRandomPoints(List<Vector3> pointCloud, int count)
    {
        HashSet<int> indices = new HashSet<int>();
        List<Vector3> sampledPoints = new List<Vector3>();
        System.Random random = new System.Random();

        while (indices.Count < count)
        {
            int randomIndex = random.Next(pointCloud.Count);
            if (!indices.Contains(randomIndex))
            {
                indices.Add(randomIndex);
                sampledPoints.Add(pointCloud[randomIndex]);
            }
        }

        return sampledPoints;
    }

    private Matrix4x4 ComputeRigidTransformation(List<Vector3> pc1, List<Vector3> pc2)
    {
        Vector3 centroid1 = ComputeCentroid(pc1);
        Vector3 centroid2 = ComputeCentroid(pc2);

        var centeredPc1 = TranslatePointCloud(pc1, -centroid1);
        var centeredPc2 = TranslatePointCloud(pc2, -centroid2);

        Matrix<double> covarianceMatrix = BuildCovarianceMatrix(centeredPc1, centeredPc2);
        var svd = covarianceMatrix.Svd();

        var rotationMatrix = svd.VT.Transpose() * svd.U.Transpose();
        if (rotationMatrix.Determinant() < 0)
        {
            var vtLastCol = svd.VT.Column(svd.VT.ColumnCount - 1).Multiply(-1);
            svd.VT.SetColumn(svd.VT.ColumnCount - 1, vtLastCol);
            rotationMatrix = svd.VT.Transpose() * svd.U.Transpose();
        }

        Vector3 translation = centroid1 - Multiply(rotationMatrix, centroid2);
        return CreateTransformationMatrix(rotationMatrix, translation);
    }

    private int CountInliers(List<Vector3> pc1, List<Vector3> pc2, Matrix4x4 transformation, float threshold)
    {
        int inliers = 0;

        foreach (var point in pc2)
        {
            Vector3 transformedPoint = transformation.MultiplyPoint(point);
            foreach (var referencePoint in pc1)
            {
                if (Vector3.Distance(transformedPoint, referencePoint) < threshold)
                {
                    inliers++;
                    break;
                }
            }
        }

        return inliers;
    }

    private Vector3 ComputeCentroid(List<Vector3> pointCloud)
    {
        Vector3 centroid = Vector3.zero;
        foreach (var point in pointCloud)
        {
            centroid += point;
        }
        return centroid / pointCloud.Count;
    }

    private List<Vector3> TranslatePointCloud(List<Vector3> pointCloud, Vector3 translation)
    {
        List<Vector3> translated = new List<Vector3>();
        foreach (var point in pointCloud)
        {
            translated.Add(point + translation);
        }
        return translated;
    }

    private Matrix<double> BuildCovarianceMatrix(List<Vector3> pc1, List<Vector3> pc2)
    {
        var covariance = Matrix<double>.Build.Dense(3, 3);

        for (int i = 0; i < pc1.Count; i++)
        {
            var p1 = Vector<double>.Build.Dense(new[] { (double)pc1[i].x, (double)pc1[i].y, (double)pc1[i].z });
            var p2 = Vector<double>.Build.Dense(new[] { (double)pc2[i].x, (double)pc2[i].y, (double)pc2[i].z });
            covariance += p1.ToColumnMatrix() * p2.ToRowMatrix();
        }

        return covariance;
    }

    private Vector3 Multiply(Matrix<double> rotation, Vector3 point)
    {
        var rotatedPoint = rotation * Vector<double>.Build.Dense(new[] { (double)point.x, (double)point.y, (double)point.z });
        return new Vector3((float)rotatedPoint[0], (float)rotatedPoint[1], (float)rotatedPoint[2]);
    }

    private Matrix4x4 CreateTransformationMatrix(Matrix<double> rotation, Vector3 translation)
    {
        Matrix4x4 transform = Matrix4x4.identity;
        transform.SetColumn(0, new Vector4((float)rotation[0, 0], (float)rotation[1, 0], (float)rotation[2, 0], 0));
        transform.SetColumn(1, new Vector4((float)rotation[0, 1], (float)rotation[1, 1], (float)rotation[2, 1], 0));
        transform.SetColumn(2, new Vector4((float)rotation[0, 2], (float)rotation[1, 2], (float)rotation[2, 2], 0));
        transform.SetColumn(3, new Vector4(translation.x, translation.y, translation.z, 1));
        return transform;
    }
    private List<Vector3> TransformPointCloud(List<Vector3> pointCloud, Matrix4x4 transformation)
    {
        List<Vector3> transformed = new List<Vector3>();
        foreach (var point in pointCloud)
        {
            transformed.Add(transformation.MultiplyPoint(point));
        }
        return transformed;
    }
    void PositionCameraForPoints(Camera camera, List<Vector3> points)
    {
        if (points == null || points.Count == 0)
        {
            Debug.LogError("No points provided for camera adjustment.");
            return;
        }

        // Compute the centroid of the points
        Vector3 centroid = Vector3.zero;
        foreach (Vector3 point in points)
        {
            centroid += point;
        }
        centroid /= points.Count;

        // Find the farthest distance from the centroid
        float maxDistance = 0f;
        foreach (Vector3 point in points)
        {
            float distance = Vector3.Distance(centroid, point);
            if (distance > maxDistance)
            {
                maxDistance = distance;
            }
        }

        // Position the camera
        camera.transform.position = centroid - camera.transform.forward * maxDistance * 2f;
        camera.transform.LookAt(centroid);

        // Adjust the field of view
        camera.fieldOfView = 60f;
        camera.nearClipPlane = 0.1f;
        camera.farClipPlane = maxDistance * 4f;
    }



}
