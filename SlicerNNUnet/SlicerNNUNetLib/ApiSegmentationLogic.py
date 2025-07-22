import os
import tempfile
from pathlib import Path
from typing import Optional

import qt
import slicer
import requests
import logging

from .Parameter import Parameter
from .Signal import Signal


class ApiSegmentationLogic:
    """
    API-based segmentation logic for nnUNet.
    
    This class implements the SegmentationLogicProtocol and handles communication
    with a remote nnUNet inference API endpoint.
    """

    def __init__(self):
        self.inferenceFinished = Signal()
        self.errorOccurred = Signal("str")
        self.progressInfo = Signal("str")
        
        self._apiBaseUrl = ""
        self._nnUNetParam = None
        self._tmpDir = qt.QTemporaryDir()
        self._currentSegmentationNode = None
        self._currentVolumeNode = None
        self._currentFileId = None  # Store the file_id from the API response

    def __del__(self):
        self.stopSegmentation()

    def setApiUrl(self, apiUrl: str):
        """Set the API base URL for inference and upload endpoints"""
        self._apiBaseUrl = apiUrl.rstrip("/")

    def getApiUrl(self) -> str:
        """Get the currently set API base URL"""
        return self._apiBaseUrl

    def getPredictUrl(self) -> str:
        """Get the prediction endpoint URL"""
        return f"{self._apiBaseUrl}/predict"
    
    def getUploadUrl(self, file_id: str) -> str:
        """Get the upload endpoint URL with the file_id"""
        return f"{self._apiBaseUrl}/upload_correction/{file_id}"

    def setParameter(self, nnUnetConf: Parameter):
        """Store parameters (required by SegmentationLogicProtocol but not used for API)"""
        self._nnUNetParam = nnUnetConf

    def startSegmentation(self, volumeNode: "slicer.vtkMRMLScalarVolumeNode") -> None:
        """Send the volume to the API for segmentation"""
        if not self._apiBaseUrl:
            self.errorOccurred("API URL is not configured.")
            return
        
        self._currentVolumeNode = volumeNode
        
        try:
            predict_url = self.getPredictUrl()
            self.progressInfo(f"Sending volume to API: {predict_url}")
            
            # Create a temporary file to store the volume
            with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Export the volume to the temporary file
            slicer.util.exportNode(volumeNode, temp_path)
            
            # Send the file to the API
            with open(temp_path, 'rb') as f:
                files = {'file': (os.path.basename(temp_path), f, 'application/octet-stream')}
                self.progressInfo("Uploading volume to API...")
                response = requests.post(predict_url, files=files)
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
            # Handle the response
            if response.status_code != 200:
                self.errorOccurred(f"API returned error: {response.status_code} - {response.text}")
                return
                
            # Get the file_id from the response headers
            self._currentFileId = response.headers.get('X-File-ID')
            if not self._currentFileId:
                self.progressInfo("Warning: No file ID received from API")
                
            # Save the response content to a temporary file
            self.nnUNetOutDir.mkdir(parents=True, exist_ok=True)
            out_file = self.nnUNetOutDir.joinpath("segmentation.nii.gz")
            
            with open(out_file, 'wb') as f:
                f.write(response.content)
                
            self.progressInfo(f"Segmentation received from API with ID: {self._currentFileId}")
            self.inferenceFinished()
        
        except Exception as e:
            self.errorOccurred(f"Error during API communication: {str(e)}")
            logging.error(f"API error: {str(e)}", exc_info=True)

    def uploadSegmentation(self, segmentationNode: "slicer.vtkMRMLSegmentationNode") -> bool:
        """Upload a corrected segmentation mask to the API"""
        if not self._apiBaseUrl:
            self.errorOccurred("API URL is not configured.")
            return False
            
        if not self._currentFileId:
            self.errorOccurred("No file ID available. Please run inference first.")
            return False
            
        try:
            upload_url = self.getUploadUrl(self._currentFileId)
            self.progressInfo(f"Uploading corrected segmentation to: {upload_url}")
            
            # Create a temporary file to store the segmentation
            with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Export segmentation to labelmap volume
            labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
                segmentationNode, labelmapVolumeNode, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY)
            
            # Export the labelmap to the temporary file
            slicer.util.exportNode(labelmapVolumeNode, temp_path)
            slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
            
            # Send the file to the API
            with open(temp_path, 'rb') as f:
                files = {'file': (os.path.basename(temp_path), f, 'application/octet-stream')}
                
                # Add metadata about the original volume if available
                data = {}
                if self._currentVolumeNode:
                    data['original_volume_name'] = self._currentVolumeNode.GetName()
                
                self.progressInfo("Uploading segmentation to API...")
                response = requests.post(upload_url, files=files, data=data)
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
            # Handle the response
            if response.status_code != 200:
                self.errorOccurred(f"API returned error: {response.status_code} - {response.text}")
                return False
                
            self.progressInfo(f"Segmentation successfully uploaded to API with ID: {self._currentFileId}")
            return True
        
        except Exception as e:
            self.errorOccurred(f"Error during upload: {str(e)}")
            logging.error(f"Upload error: {str(e)}", exc_info=True)
            return False

    def stopSegmentation(self):
        """Stop the current segmentation process"""
        # For API-based segmentation, we don't need to do anything special
        pass

    def waitForSegmentationFinished(self):
        """Wait for the segmentation to finish"""
        # For API-based segmentation, we don't need to do anything special
        pass

    def loadSegmentation(self) -> "slicer.vtkMRMLSegmentationNode":
        """Load the segmentation result from the API response"""
        try:
            segmentation_file = self.nnUNetOutDir.joinpath("segmentation.nii.gz")
            
            if not segmentation_file.exists():
                raise RuntimeError("Segmentation file not found. API segmentation may have failed.")
                
            # Load as labelmap volume first
            labelmapVolumeNode = slicer.util.loadLabelVolume(str(segmentation_file))
            
            # Convert to segmentation
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            segmentationNode.SetName(f"{self._currentVolumeNode.GetName()}_ApiSegmentation")
            
            # Import the labelmap to the segmentation
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                labelmapVolumeNode, segmentationNode)
            
            # Clean up the labelmap volume
            slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
            
            # Store the current segmentation node for potential upload
            self._currentSegmentationNode = segmentationNode
            
            return segmentationNode
        
        except Exception as e:
            raise RuntimeError(f"Failed to load segmentation: {str(e)}")
    
    @property
    def nnUNetOutDir(self):
        return Path(self._tmpDir.path()).joinpath("output")
    
    def getCurrentSegmentation(self) -> Optional["slicer.vtkMRMLSegmentationNode"]:
        """Get the current segmentation node"""
        return self._currentSegmentationNode
