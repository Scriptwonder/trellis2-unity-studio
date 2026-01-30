using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using Trellis2;

/// <summary>
/// Demo script showing how to use the Trellis2Client.
/// Attach this to a GameObject along with Trellis2Client.
/// </summary>
public class Trellis2Demo : MonoBehaviour
{
    [Header("References")]
    public Trellis2Client trellis2Client;

    [Header("UI (Optional)")]
    public InputField promptInput;
    public Button generateButton;
    public Button generateFromImageButton;
    public Text statusText;
    public RawImage previewImage;

    [Header("GLB Loading (Optional)")]
    [Tooltip("Parent transform for spawned 3D models")]
    public Transform modelParent;

    [Header("Test Settings")]
    public string testPrompt = "A cute robot toy";
    public Texture2D testImage;

    private bool isGenerating = false;

    private void Start()
    {
        // Get client reference if not assigned
        if (trellis2Client == null)
        {
            trellis2Client = GetComponent<Trellis2Client>();
        }

        // Subscribe to events
        if (trellis2Client != null)
        {
            trellis2Client.OnGenerationComplete += OnGenerationComplete;
            trellis2Client.OnGenerationFailed += OnGenerationFailed;
            trellis2Client.OnGenerationProgress += OnGenerationProgress;
        }

        // Setup UI buttons
        if (generateButton != null)
        {
            generateButton.onClick.AddListener(OnGenerateButtonClicked);
        }

        if (generateFromImageButton != null)
        {
            generateFromImageButton.onClick.AddListener(OnGenerateFromImageClicked);
        }

        UpdateStatus("Ready");
    }

    private void OnDestroy()
    {
        if (trellis2Client != null)
        {
            trellis2Client.OnGenerationComplete -= OnGenerationComplete;
            trellis2Client.OnGenerationFailed -= OnGenerationFailed;
            trellis2Client.OnGenerationProgress -= OnGenerationProgress;
        }
    }

    #region UI Handlers

    private void OnGenerateButtonClicked()
    {
        string prompt = promptInput != null ? promptInput.text : testPrompt;

        if (string.IsNullOrWhiteSpace(prompt))
        {
            UpdateStatus("Error: Please enter a prompt");
            return;
        }

        GenerateFromText(prompt);
    }

    private void OnGenerateFromImageClicked()
    {
        if (testImage != null)
        {
            GenerateFromTexture(testImage);
        }
        else
        {
            UpdateStatus("Error: No test image assigned");
        }
    }

    #endregion

    #region Public API Examples

    /// <summary>
    /// Example: Generate 3D model from text prompt.
    /// </summary>
    public void GenerateFromText(string prompt)
    {
        if (isGenerating)
        {
            Debug.LogWarning("[Demo] Generation already in progress");
            return;
        }

        isGenerating = true;
        UpdateStatus($"Generating: {prompt}");

        // Method 1: Using callback
        trellis2Client.GenerateFromText(prompt, OnResultReceived);

        // Method 2: Using coroutine (alternative)
        // StartCoroutine(GenerateWithCoroutine(prompt));
    }

    /// <summary>
    /// Example: Generate 3D model from Texture2D.
    /// </summary>
    public void GenerateFromTexture(Texture2D texture)
    {
        if (isGenerating)
        {
            Debug.LogWarning("[Demo] Generation already in progress");
            return;
        }

        isGenerating = true;
        UpdateStatus("Generating from image...");

        trellis2Client.GenerateFromTexture(texture, OnResultReceived);
    }

    /// <summary>
    /// Example: Generate 3D model from image file.
    /// </summary>
    public void GenerateFromImageFile(string filePath)
    {
        if (isGenerating)
        {
            Debug.LogWarning("[Demo] Generation already in progress");
            return;
        }

        isGenerating = true;
        UpdateStatus($"Generating from: {filePath}");

        trellis2Client.GenerateFromImageFile(filePath, OnResultReceived);
    }

    /// <summary>
    /// Example: Using coroutine directly for more control.
    /// </summary>
    private IEnumerator GenerateWithCoroutine(string prompt)
    {
        UpdateStatus($"Starting generation: {prompt}");

        var result = new GenerationResult();

        yield return trellis2Client.GenerateFromTextCoroutine(prompt, (r) => result = r);

        if (string.IsNullOrEmpty(result.error))
        {
            Debug.Log($"[Demo] GLB saved to: {result.localGlbPath}");

            // Load the GLB (requires GLTFUtility or similar)
            // LoadGLB(result.localGlbPath);
        }
        else
        {
            Debug.LogError($"[Demo] Failed: {result.error}");
        }

        isGenerating = false;
    }

    #endregion

    #region Event Handlers

    private void OnResultReceived(GenerationResult result)
    {
        isGenerating = false;

        if (!string.IsNullOrEmpty(result.error))
        {
            UpdateStatus($"Error: {result.error}");
            return;
        }

        UpdateStatus($"Complete! GLB: {result.localGlbPath}");

        // Load preview image if available
        if (!string.IsNullOrEmpty(result.localImagePath) && previewImage != null)
        {
            StartCoroutine(LoadPreviewImage(result.localImagePath));
        }

        // Load GLB model (requires GLTFUtility, UnityGLTF, or similar package)
        // LoadGLB(result.localGlbPath);
    }

    private void OnGenerationComplete(GenerationResult result)
    {
        Debug.Log($"[Demo] Generation complete: {result.jobId}");
    }

    private void OnGenerationFailed(GenerationResult result)
    {
        Debug.LogError($"[Demo] Generation failed: {result.error}");
    }

    private void OnGenerationProgress(string jobId, float elapsedTime, string stage)
    {
        UpdateStatus($"{stage} ({elapsedTime:F0}s)");
    }

    #endregion

    #region Helpers

    private void UpdateStatus(string message)
    {
        Debug.Log($"[Demo] {message}");

        if (statusText != null)
        {
            statusText.text = message;
        }
    }

    private IEnumerator LoadPreviewImage(string imagePath)
    {
        byte[] imageData = System.IO.File.ReadAllBytes(imagePath);
        Texture2D texture = new Texture2D(2, 2);
        texture.LoadImage(imageData);
        previewImage.texture = texture;
        yield return null;
    }

    /// <summary>
    /// Load a GLB file and instantiate it in the scene.
    /// Requires a GLB loading package like GLTFUtility or UnityGLTF.
    /// </summary>
    private void LoadGLB(string glbPath)
    {
        // Example using GLTFUtility (https://github.com/Siccity/GLTFUtility):
        // GameObject model = Siccity.GLTFUtility.Importer.LoadFromFile(glbPath);
        // if (modelParent != null) model.transform.SetParent(modelParent);

        // Example using UnityGLTF:
        // var loader = new UnityGLTF.GLTFSceneImporter(glbPath, new UnityGLTF.ImportOptions());
        // StartCoroutine(loader.LoadScene());

        Debug.Log($"[Demo] GLB loading not implemented. File saved at: {glbPath}");
        Debug.Log("[Demo] Install GLTFUtility or UnityGLTF to load GLB files at runtime.");
    }

    #endregion

    #region Context Menu (Editor Testing)

#if UNITY_EDITOR
    [ContextMenu("Test: Generate from Text")]
    private void TestGenerateFromText()
    {
        GenerateFromText(testPrompt);
    }

    [ContextMenu("Test: Generate from Image")]
    private void TestGenerateFromImage()
    {
        if (testImage != null)
        {
            GenerateFromTexture(testImage);
        }
        else
        {
            Debug.LogError("[Demo] No test image assigned");
        }
    }

    [ContextMenu("Test: Check Server Health")]
    private void TestCheckHealth()
    {
        trellis2Client.CheckHealth((healthy) =>
        {
            Debug.Log($"[Demo] Server health: {(healthy ? "OK" : "FAILED")}");
        });
    }
#endif

    #endregion
}
