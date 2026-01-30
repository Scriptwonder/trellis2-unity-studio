using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace Trellis2
{
    /// <summary>
    /// Quality presets for 3D generation.
    /// </summary>
    public enum GenerationQuality
    {
        SuperFast,  // ~15s, lowest quality, for real-time iteration
        Fast,       // ~60s, lower quality
        Balanced,   // ~90s, good balance
        High        // ~180s, best quality
    }

    /// <summary>
    /// Result of a generation job.
    /// </summary>
    [Serializable]
    public class GenerationResult
    {
        public string jobId;
        public string status;
        public string glbUrl;
        public string imageUrl;
        public string localGlbPath;
        public string localImagePath;
        public string error;
        public Dictionary<string, float> timings;
    }

    /// <summary>
    /// Unity client for TRELLIS.2 API.
    /// Supports text-to-3D and image-to-3D generation.
    /// </summary>
    public class Trellis2Client : MonoBehaviour
    {
        [Header("Server Configuration")]
        [Tooltip("Base URL of the TRELLIS.2 API server")]
        public string serverUrl = "http://localhost:8000";

        [Header("Generation Settings")]
        [Tooltip("Quality preset for generation")]
        public GenerationQuality quality = GenerationQuality.Balanced;

        [Tooltip("Random seed for reproducibility (use -1 for random)")]
        public int seed = 42;

        [Header("Download Settings")]
        [Tooltip("Directory to save downloaded files (relative to Application.persistentDataPath)")]
        public string downloadDirectory = "Trellis2Downloads";

        [Tooltip("Polling interval in seconds when waiting for job completion")]
        public float pollInterval = 2f;

        public event Action<GenerationResult> OnGenerationComplete;
        public event Action<GenerationResult> OnGenerationFailed;
        /// <summary>Params: jobId, elapsedTime, stageDescription</summary>
        public event Action<string, float, string> OnGenerationProgress;

        private string DownloadPath => Path.Combine(Application.persistentDataPath, downloadDirectory);

        private void Awake()
        {
            // Ensure download directory exists
            if (!Directory.Exists(DownloadPath))
            {
                Directory.CreateDirectory(DownloadPath);
            }
        }

        #region Public API

        /// <summary>
        /// Generate a 3D model from a text prompt.
        /// </summary>
        /// <param name="prompt">Text description of the 3D object</param>
        /// <param name="callback">Optional callback when complete</param>
        public void GenerateFromText(string prompt, Action<GenerationResult> callback = null)
        {
            StartCoroutine(GenerateFromTextCoroutine(prompt, callback));
        }

        /// <summary>
        /// Generate a 3D model from a text prompt (coroutine version).
        /// </summary>
        public IEnumerator GenerateFromTextCoroutine(string prompt, Action<GenerationResult> callback = null)
        {
            var result = new GenerationResult();

            // Submit job
            Debug.Log($"[Trellis2] Submitting text-to-3D job: {prompt}");
            yield return SubmitTextJob(prompt, result);

            if (!string.IsNullOrEmpty(result.error))
            {
                HandleError(result, callback);
                yield break;
            }

            // Wait for completion
            yield return WaitForJobCompletion(result);

            if (!string.IsNullOrEmpty(result.error))
            {
                HandleError(result, callback);
                yield break;
            }

            // Download files
            yield return DownloadResults(result);

            // Success
            HandleSuccess(result, callback);
        }

        /// <summary>
        /// Generate a 3D model from a Texture2D.
        /// </summary>
        /// <param name="texture">Input texture</param>
        /// <param name="callback">Optional callback when complete</param>
        public void GenerateFromTexture(Texture2D texture, Action<GenerationResult> callback = null)
        {
            StartCoroutine(GenerateFromTextureCoroutine(texture, callback));
        }

        /// <summary>
        /// Generate a 3D model from a Texture2D (coroutine version).
        /// </summary>
        public IEnumerator GenerateFromTextureCoroutine(Texture2D texture, Action<GenerationResult> callback = null)
        {
            var result = new GenerationResult();

            // Convert texture to PNG bytes
            byte[] imageData = texture.EncodeToPNG();

            // Submit job
            Debug.Log("[Trellis2] Submitting image-to-3D job");
            yield return SubmitImageJob(imageData, "input.png", result);

            if (!string.IsNullOrEmpty(result.error))
            {
                HandleError(result, callback);
                yield break;
            }

            // Wait for completion
            yield return WaitForJobCompletion(result);

            if (!string.IsNullOrEmpty(result.error))
            {
                HandleError(result, callback);
                yield break;
            }

            // Download files
            yield return DownloadResults(result);

            // Success
            HandleSuccess(result, callback);
        }

        /// <summary>
        /// Generate a 3D model from an image file path.
        /// </summary>
        /// <param name="imagePath">Path to the image file</param>
        /// <param name="callback">Optional callback when complete</param>
        public void GenerateFromImageFile(string imagePath, Action<GenerationResult> callback = null)
        {
            StartCoroutine(GenerateFromImageFileCoroutine(imagePath, callback));
        }

        /// <summary>
        /// Generate a 3D model from an image file (coroutine version).
        /// </summary>
        public IEnumerator GenerateFromImageFileCoroutine(string imagePath, Action<GenerationResult> callback = null)
        {
            var result = new GenerationResult();

            if (!File.Exists(imagePath))
            {
                result.error = $"Image file not found: {imagePath}";
                HandleError(result, callback);
                yield break;
            }

            byte[] imageData = File.ReadAllBytes(imagePath);
            string fileName = Path.GetFileName(imagePath);

            // Submit job
            Debug.Log($"[Trellis2] Submitting image-to-3D job: {fileName}");
            yield return SubmitImageJob(imageData, fileName, result);

            if (!string.IsNullOrEmpty(result.error))
            {
                HandleError(result, callback);
                yield break;
            }

            // Wait for completion
            yield return WaitForJobCompletion(result);

            if (!string.IsNullOrEmpty(result.error))
            {
                HandleError(result, callback);
                yield break;
            }

            // Download files
            yield return DownloadResults(result);

            // Success
            HandleSuccess(result, callback);
        }

        /// <summary>
        /// Check if the server is healthy.
        /// </summary>
        public void CheckHealth(Action<bool> callback)
        {
            StartCoroutine(CheckHealthCoroutine(callback));
        }

        /// <summary>
        /// Check server health (coroutine version).
        /// </summary>
        public IEnumerator CheckHealthCoroutine(Action<bool> callback)
        {
            using (var request = UnityWebRequest.Get($"{serverUrl}/health"))
            {
                yield return request.SendWebRequest();
                callback?.Invoke(request.result == UnityWebRequest.Result.Success);
            }
        }

        #endregion

        #region Private Methods

        private IEnumerator SubmitTextJob(string prompt, GenerationResult result)
        {
            var requestBody = new TextSubmitRequest
            {
                prompt = prompt,
                quality = GetQualityString(),
                seed = seed >= 0 ? seed : UnityEngine.Random.Range(0, int.MaxValue)
            };

            string json = JsonUtility.ToJson(requestBody);

            using (var request = new UnityWebRequest($"{serverUrl}/submit/text", "POST"))
            {
                byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
                request.uploadHandler = new UploadHandlerRaw(bodyRaw);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");

                yield return request.SendWebRequest();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    result.error = $"Failed to submit job: {request.error}";
                    yield break;
                }

                var response = JsonUtility.FromJson<JobSubmitResponse>(request.downloadHandler.text);
                result.jobId = response.job_id;
                result.status = response.status;
            }
        }

        private IEnumerator SubmitImageJob(byte[] imageData, string fileName, GenerationResult result)
        {
            var form = new List<IMultipartFormSection>
            {
                new MultipartFormFileSection("file", imageData, fileName, "image/png")
            };

            string url = $"{serverUrl}/submit/image?quality={GetQualityString()}&seed={seed}";

            using (var request = UnityWebRequest.Post(url, form))
            {
                yield return request.SendWebRequest();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    result.error = $"Failed to submit job: {request.error}";
                    yield break;
                }

                var response = JsonUtility.FromJson<JobSubmitResponse>(request.downloadHandler.text);
                result.jobId = response.job_id;
                result.status = response.status;
            }
        }

        private IEnumerator WaitForJobCompletion(GenerationResult result)
        {
            float startTime = Time.time;

            while (true)
            {
                using (var request = UnityWebRequest.Get($"{serverUrl}/status/{result.jobId}"))
                {
                    yield return request.SendWebRequest();

                    if (request.result != UnityWebRequest.Result.Success)
                    {
                        result.error = $"Failed to check status: {request.error}";
                        yield break;
                    }

                    var status = JsonUtility.FromJson<JobStatusResponse>(request.downloadHandler.text);
                    result.status = status.status;

                    float elapsed = Time.time - startTime;
                    string stageInfo = status.stage_description ?? status.status;
                    OnGenerationProgress?.Invoke(result.jobId, elapsed, stageInfo);
                    Debug.Log($"[Trellis2] {stageInfo} ({elapsed:F0}s)");

                    if (status.status == "done")
                    {
                        // Parse result URLs
                        if (status.result != null)
                        {
                            result.glbUrl = status.result.glb;
                            result.imageUrl = status.result.image;
                        }
                        Debug.Log($"[Trellis2] Job completed in {elapsed:F1}s");
                        yield break;
                    }
                    else if (status.status == "failed")
                    {
                        result.error = status.error ?? "Job failed with unknown error";
                        yield break;
                    }
                }

                yield return new WaitForSeconds(pollInterval);
            }
        }

        private IEnumerator DownloadResults(GenerationResult result)
        {
            // Download GLB
            if (!string.IsNullOrEmpty(result.glbUrl))
            {
                string glbPath = Path.Combine(DownloadPath, $"{result.jobId}.glb");
                yield return DownloadFile($"{serverUrl}/{result.glbUrl}", glbPath);
                result.localGlbPath = glbPath;
                Debug.Log($"[Trellis2] Downloaded GLB: {glbPath}");
            }

            // Download image (if available, for text-to-3D)
            if (!string.IsNullOrEmpty(result.imageUrl))
            {
                string imagePath = Path.Combine(DownloadPath, $"{result.jobId}_image.png");
                yield return DownloadFile($"{serverUrl}/{result.imageUrl}", imagePath);
                result.localImagePath = imagePath;
                Debug.Log($"[Trellis2] Downloaded image: {imagePath}");
            }
        }

        private IEnumerator DownloadFile(string url, string localPath)
        {
            using (var request = UnityWebRequest.Get(url))
            {
                request.downloadHandler = new DownloadHandlerFile(localPath);
                yield return request.SendWebRequest();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogError($"[Trellis2] Failed to download {url}: {request.error}");
                }
            }
        }

        private void HandleSuccess(GenerationResult result, Action<GenerationResult> callback)
        {
            Debug.Log($"[Trellis2] Generation complete: {result.localGlbPath}");
            callback?.Invoke(result);
            OnGenerationComplete?.Invoke(result);
        }

        private void HandleError(GenerationResult result, Action<GenerationResult> callback)
        {
            Debug.LogError($"[Trellis2] Generation failed: {result.error}");
            callback?.Invoke(result);
            OnGenerationFailed?.Invoke(result);
        }

        private string GetQualityString()
        {
            return quality switch
            {
                GenerationQuality.SuperFast => "superfast",
                GenerationQuality.Fast => "fast",
                GenerationQuality.High => "high",
                _ => "balanced"
            };
        }

        #endregion

        #region JSON Data Classes

        [Serializable]
        private class TextSubmitRequest
        {
            public string prompt;
            public string quality;
            public int seed;
        }

        [Serializable]
        private class JobSubmitResponse
        {
            public string job_id;
            public string status;
        }

        [Serializable]
        private class JobStatusResponse
        {
            public string job_id;
            public string type;
            public string status;
            public string stage;
            public string stage_description;
            public string error;
            public JobResult result;
        }

        [Serializable]
        private class JobResult
        {
            public string glb;
            public string image;
        }

        #endregion
    }
}
