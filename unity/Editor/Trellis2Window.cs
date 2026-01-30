using UnityEngine;
using UnityEditor;
using System.Collections;
using System.Collections.Generic;
using Trellis2;

/// <summary>
/// Editor window for testing TRELLIS.2 generation directly in the Unity Editor.
/// Supports multiple concurrent jobs.
/// </summary>
public class Trellis2Window : EditorWindow
{
    private string serverUrl = "http://localhost:8000";
    private string prompt = "A cute robot toy";
    private GenerationQuality quality = GenerationQuality.Balanced;
    private int seed = 42;
    private Texture2D inputImage;

    private bool autoAddToScene = true;

    private readonly List<JobEntry> jobs = new List<JobEntry>();

    private Vector2 scrollPosition;
    private Vector2 jobsScrollPosition;

    [MenuItem("Tools/TRELLIS.2/Generation Window")]
    public static void ShowWindow()
    {
        var window = GetWindow<Trellis2Window>("TRELLIS.2");
        window.minSize = new Vector2(400, 500);
    }

    private void OnGUI()
    {
        scrollPosition = EditorGUILayout.BeginScrollView(scrollPosition);

        // Header
        EditorGUILayout.LabelField("TRELLIS.2 Generator", EditorStyles.boldLabel);
        EditorGUILayout.Space();

        // Server Settings
        EditorGUILayout.LabelField("Server", EditorStyles.boldLabel);
        serverUrl = EditorGUILayout.TextField("Server URL", serverUrl);

        EditorGUILayout.Space();

        // Generation Settings
        EditorGUILayout.LabelField("Settings", EditorStyles.boldLabel);
        quality = (GenerationQuality)EditorGUILayout.EnumPopup("Quality", quality);
        seed = EditorGUILayout.IntField("Seed (-1 for random)", seed);
        autoAddToScene = EditorGUILayout.Toggle("Auto-Add to Scene", autoAddToScene);

        EditorGUILayout.Space();

        // Text-to-3D Section
        EditorGUILayout.LabelField("Text to 3D", EditorStyles.boldLabel);
        prompt = EditorGUILayout.TextField("Prompt", prompt);

        EditorGUI.BeginDisabledGroup(string.IsNullOrWhiteSpace(prompt));
        if (GUILayout.Button("Generate from Text", GUILayout.Height(30)))
        {
            GenerateFromText();
        }
        EditorGUI.EndDisabledGroup();

        EditorGUILayout.Space();

        // Image-to-3D Section
        EditorGUILayout.LabelField("Image to 3D", EditorStyles.boldLabel);
        inputImage = (Texture2D)EditorGUILayout.ObjectField("Input Image", inputImage, typeof(Texture2D), false);

        EditorGUI.BeginDisabledGroup(inputImage == null);
        if (GUILayout.Button("Generate from Image", GUILayout.Height(30)))
        {
            GenerateFromImage();
        }
        EditorGUI.EndDisabledGroup();

        EditorGUILayout.Space();

        // Jobs list
        DrawJobsList();

        EditorGUILayout.EndScrollView();
    }

    private void DrawJobsList()
    {
        if (jobs.Count == 0)
            return;

        EditorGUILayout.LabelField("Jobs", EditorStyles.boldLabel);

        bool hasCompleted = false;

        for (int i = jobs.Count - 1; i >= 0; i--)
        {
            var job = jobs[i];
            if (!job.isRunning) hasCompleted = true;

            EditorGUILayout.BeginVertical(EditorStyles.helpBox);

            // Job header: label + status
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(job.label, EditorStyles.boldLabel, GUILayout.MaxWidth(200));
            GUILayout.FlexibleSpace();
            EditorGUILayout.LabelField(job.status, GUILayout.MaxWidth(250));
            EditorGUILayout.EndHorizontal();

            // Result actions for completed jobs
            if (!job.isRunning && !string.IsNullOrEmpty(job.assetPath))
            {
                EditorGUILayout.BeginHorizontal();
                if (GUILayout.Button("Select in Project", GUILayout.Height(20)))
                {
                    var asset = AssetDatabase.LoadAssetAtPath<Object>(job.assetPath);
                    if (asset != null)
                    {
                        Selection.activeObject = asset;
                        EditorGUIUtility.PingObject(asset);
                    }
                }
                if (GUILayout.Button("Add to Scene", GUILayout.Height(20)))
                {
                    InstantiateInScene(job);
                }
                if (GUILayout.Button("Open in Explorer", GUILayout.Height(20)))
                {
                    EditorUtility.RevealInFinder(job.glbPath);
                }
                EditorGUILayout.EndHorizontal();
            }

            // Preview image
            if (job.generatedImage != null)
            {
                float aspectRatio = (float)job.generatedImage.height / job.generatedImage.width;
                float width = EditorGUIUtility.currentViewWidth - 60;
                float height = width * aspectRatio;
                Rect rect = GUILayoutUtility.GetRect(width, Mathf.Min(height, 200));
                EditorGUI.DrawPreviewTexture(rect, job.generatedImage, null, ScaleMode.ScaleToFit);
            }

            EditorGUILayout.EndVertical();
        }

        if (hasCompleted)
        {
            EditorGUILayout.Space();
            if (GUILayout.Button("Clear Completed"))
            {
                jobs.RemoveAll(j => !j.isRunning);
                Repaint();
            }
        }
    }

    private void GenerateFromText()
    {
        var job = new JobEntry
        {
            label = prompt.Length > 30 ? prompt.Substring(0, 30) + "..." : prompt,
            status = "Submitting...",
            isRunning = true
        };
        jobs.Add(job);
        Repaint();

        EditorCoroutine.Start(GenerateFromTextCoroutine(job));
    }

    private void GenerateFromImage()
    {
        var job = new JobEntry
        {
            label = "Image: " + inputImage.name,
            status = "Submitting...",
            isRunning = true
        };
        jobs.Add(job);
        Repaint();

        EditorCoroutine.Start(GenerateFromImageCoroutine(job, inputImage));
    }

    private IEnumerator GenerateFromTextCoroutine(JobEntry job)
    {
        string submitUrl = $"{serverUrl}/submit/text";
        string json = JsonUtility.ToJson(new TextRequest { prompt = prompt, quality = GetQualityString(), seed = seed });

        using (var www = new UnityEngine.Networking.UnityWebRequest(submitUrl, "POST"))
        {
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);
            www.uploadHandler = new UnityEngine.Networking.UploadHandlerRaw(bodyRaw);
            www.downloadHandler = new UnityEngine.Networking.DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");

            yield return www.SendWebRequest();

            if (www.result != UnityEngine.Networking.UnityWebRequest.Result.Success)
            {
                job.status = $"Error: {www.error}";
                job.isRunning = false;
                Repaint();
                yield break;
            }

            var response = JsonUtility.FromJson<JobResponse>(www.downloadHandler.text);
            job.jobId = response.job_id;
            yield return WaitForJob(job);
        }
    }

    private IEnumerator GenerateFromImageCoroutine(JobEntry job, Texture2D image)
    {
        byte[] imageData = image.EncodeToPNG();
        string url = $"{serverUrl}/submit/image?quality={GetQualityString()}&seed={seed}";

        var form = new List<UnityEngine.Networking.IMultipartFormSection>
        {
            new UnityEngine.Networking.MultipartFormFileSection("file", imageData, "input.png", "image/png")
        };

        using (var www = UnityEngine.Networking.UnityWebRequest.Post(url, form))
        {
            yield return www.SendWebRequest();

            if (www.result != UnityEngine.Networking.UnityWebRequest.Result.Success)
            {
                job.status = $"Error: {www.error}";
                job.isRunning = false;
                Repaint();
                yield break;
            }

            var response = JsonUtility.FromJson<JobResponse>(www.downloadHandler.text);
            job.jobId = response.job_id;
            yield return WaitForJob(job);
        }
    }

    private IEnumerator WaitForJob(JobEntry job)
    {
        job.status = $"Processing ({job.jobId})";
        Repaint();

        float startTime = (float)EditorApplication.timeSinceStartup;

        while (true)
        {
            using (var www = UnityEngine.Networking.UnityWebRequest.Get($"{serverUrl}/status/{job.jobId}"))
            {
                yield return www.SendWebRequest();

                if (www.result != UnityEngine.Networking.UnityWebRequest.Result.Success)
                {
                    job.status = $"Error: {www.error}";
                    job.isRunning = false;
                    Repaint();
                    yield break;
                }

                var statusResponse = JsonUtility.FromJson<StatusResponse>(www.downloadHandler.text);
                float elapsed = (float)EditorApplication.timeSinceStartup - startTime;

                if (statusResponse.status == "done")
                {
                    job.status = $"Complete ({elapsed:F1}s)";

                    if (statusResponse.result != null && !string.IsNullOrEmpty(statusResponse.result.glb))
                    {
                        yield return DownloadFile($"{serverUrl}/{statusResponse.result.glb}", job);
                    }

                    if (statusResponse.result != null && !string.IsNullOrEmpty(statusResponse.result.image))
                    {
                        yield return DownloadImage($"{serverUrl}/{statusResponse.result.image}", job);
                    }

                    job.isRunning = false;
                    Repaint();
                    yield break;
                }
                else if (statusResponse.status == "failed")
                {
                    job.status = $"Failed: {statusResponse.error}";
                    job.isRunning = false;
                    Repaint();
                    yield break;
                }
                else
                {
                    job.status = $"Processing... ({elapsed:F0}s)";
                    Repaint();
                }
            }

            yield return new EditorWaitForSeconds(2f);
        }
    }

    private IEnumerator DownloadFile(string url, JobEntry job)
    {
        string localDirectory = System.IO.Path.Combine(Application.dataPath, "Trellis2Results");
        if (!System.IO.Directory.Exists(localDirectory))
        {
            System.IO.Directory.CreateDirectory(localDirectory);
        }

        string baseName = System.IO.Path.GetFileName(url);
        if (string.IsNullOrEmpty(baseName) || !baseName.EndsWith(".glb", System.StringComparison.OrdinalIgnoreCase))
        {
            baseName = $"{job.jobId}.glb";
        }

        string downloadPath = System.IO.Path.Combine(localDirectory, baseName);

        using (var www = UnityEngine.Networking.UnityWebRequest.Get(url))
        {
            www.downloadHandler = new UnityEngine.Networking.DownloadHandlerFile(downloadPath);
            yield return www.SendWebRequest();

            if (www.result == UnityEngine.Networking.UnityWebRequest.Result.Success)
            {
                job.glbPath = downloadPath;

                string relativePath = "Assets/Trellis2Results/" + baseName;
                AssetDatabase.ImportAsset(relativePath, ImportAssetOptions.ForceUpdate);
                job.assetPath = relativePath;

                Debug.Log($"[Trellis2] Downloaded to project: {relativePath}");

                if (autoAddToScene)
                {
                    InstantiateInScene(job);
                }
            }
        }
    }

    private IEnumerator DownloadImage(string url, JobEntry job)
    {
        using (var www = UnityEngine.Networking.UnityWebRequestTexture.GetTexture(url))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityEngine.Networking.UnityWebRequest.Result.Success)
            {
                job.generatedImage = UnityEngine.Networking.DownloadHandlerTexture.GetContent(www);
            }
        }
    }

    private void InstantiateInScene(JobEntry job)
    {
        if (string.IsNullOrEmpty(job.assetPath))
            return;

        var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(job.assetPath);
        if (prefab == null)
        {
            Debug.LogWarning($"[Trellis2] Could not load {job.assetPath} as GameObject. " +
                "Ensure a GLB/glTF importer package is installed (GLTFUtility or UnityGLTF).");
            return;
        }

        var instance = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
        instance.name = !string.IsNullOrEmpty(job.label) ? job.label : prefab.name;
        instance.transform.position = Vector3.zero;
        Undo.RegisterCreatedObjectUndo(instance, "Trellis2 Add to Scene");
        Selection.activeGameObject = instance;
        EditorGUIUtility.PingObject(instance);
    }

    private string GetQualityString()
    {
        return quality switch
        {
            GenerationQuality.Fast => "fast",
            GenerationQuality.High => "high",
            _ => "balanced"
        };
    }

    #region Inner Types

    private class JobEntry
    {
        public string jobId;
        public string label;
        public string status;
        public bool isRunning;
        public string glbPath;
        public string assetPath;
        public Texture2D generatedImage;
    }

    [System.Serializable]
    private class TextRequest
    {
        public string prompt;
        public string quality;
        public int seed;
    }

    [System.Serializable]
    private class JobResponse
    {
        public string job_id;
        public string status;
    }

    [System.Serializable]
    private class StatusResponse
    {
        public string job_id;
        public string status;
        public string error;
        public ResultData result;
    }

    [System.Serializable]
    private class ResultData
    {
        public string glb;
        public string image;
    }

    #endregion
}

/// <summary>
/// Editor coroutine helper with support for nested IEnumerator, AsyncOperation,
/// and CustomYieldInstruction yields.
/// </summary>
public static class EditorCoroutine
{
    public static void Start(IEnumerator routine)
    {
        var stack = new System.Collections.Generic.Stack<IEnumerator>();
        stack.Push(routine);

        EditorApplication.CallbackFunction callback = null;
        object current = null;
        bool needsMove = true;

        callback = () =>
        {
            if (stack.Count == 0)
            {
                EditorApplication.update -= callback;
                return;
            }

            // If waiting on a yield, check if it's done
            if (!needsMove)
            {
                if (current is AsyncOperation asyncOp && !asyncOp.isDone)
                    return;
                if (current is CustomYieldInstruction customYield && customYield.keepWaiting)
                    return;
                needsMove = true;
            }

            if (needsMove)
            {
                var active = stack.Peek();
                if (!active.MoveNext())
                {
                    // Current coroutine finished — pop and resume parent
                    stack.Pop();
                    if (stack.Count == 0)
                    {
                        EditorApplication.update -= callback;
                        return;
                    }
                    // Let the parent advance on the next frame
                    needsMove = true;
                    return;
                }

                current = active.Current;

                // If the yielded value is a nested coroutine, push it onto the stack
                if (current is IEnumerator nested)
                {
                    stack.Push(nested);
                    needsMove = true;
                    return;
                }

                needsMove = false;
            }
        };
        EditorApplication.update += callback;
    }
}

public class EditorWaitForSeconds : CustomYieldInstruction
{
    private double targetTime;

    public EditorWaitForSeconds(float seconds)
    {
        targetTime = EditorApplication.timeSinceStartup + seconds;
    }

    public override bool keepWaiting => EditorApplication.timeSinceStartup < targetTime;
}
