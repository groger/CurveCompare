#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <string>
#include <iomanip>

// VTK include
#include <vtkPolyDataReader.h>
#include <vtkXMLPolyDataReader.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkPointData.h>
#include <vtkSpline.h>
#include <vtkParametricSpline.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkSplineRepresentation.h>
#include <vtkMath.h>
#include <vtkParametricFunctionSource.h>
#include <vtkCardinalSpline.h>
#include <vtkSCurveSpline.h>
#include <vtkObject.h>
#include <vtkPolyLine.h>

// ITK include
#include <itkImage.h>
#include <itkVectorContainer.h>
#include <itkVector.h>
#include <itkDiffusionTensor3D.h>
#include <itkImageAdaptor.h>
#include <itkPoint.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkContinuousIndex.h>
#include <itkSpatialObject.h>
#include <itkLabelOverlapMeasuresImageFilter.h>
#include <itkKappaStatisticImageToImageMetric.h>
#include "itkTranslationTransform.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkImageRegionIterator.h"

// VNL Includes
#include <vnl/vnl_matrix.h>
#include <vnl/vnl_vector_fixed.h>

#include <CurveCompareCLP.h>

using namespace std;

std::vector<double> FillOverlapTable(vtkSmartPointer<vtkPolyData> Source, vtkSmartPointer<vtkPolyData> Target,string   ReferenceScalarVolume,int voxelLabel)
{  std::cout<<"Overlap calculating..."<<std::endl;
  std::vector<double> OverlapTable;
  vtkPoints* SourcePoints = Source->GetPoints();
  vtkPoints* TargetPoints = Target->GetPoints();
  int numberPointsSource=SourcePoints->GetNumberOfPoints();
  int numberPointsTarget=TargetPoints->GetNumberOfPoints();

  const unsigned int DIM = 3;
  typedef unsigned short ScalarPixelType;
  typedef itk::Image<ScalarPixelType, DIM> IntImageType;
  /*typedef itk::DiffusionTensor3D<double> TensorPixelType;
  typedef itk::Image<TensorPixelType, DIM> TensorImageType;
  // Setup tensor file if available
  typedef itk::ImageFileReader<TensorImageType> TensorImageReader;
  TensorImageReader::Pointer tensorreader = NULL;
  tensorreader = TensorImageReader::New();   
  tensorreader->SetFileName(tensorVolume.c_str());
  tensorreader->Update();*/
  typedef itk::ImageFileReader<IntImageType> ScalarImageReader;
  ScalarImageReader::Pointer scalarReader = NULL;
  scalarReader = ScalarImageReader::New();   
  scalarReader->SetFileName(ReferenceScalarVolume.c_str());
  scalarReader->Update();

  // Need to allocate two images to write into for creating
  // the fiber label maps
  IntImageType::Pointer labelimage1;//classic voxelization of Source
  IntImageType::Pointer labelimage2;//classic voxelization of Target
  IntImageType::Pointer labelimageCountF;//voxelization of Source by counting how many fibers for each voxel
  IntImageType::Pointer labelimage2CountF;//voxelization of Target by counting how many fibers for each voxel

  labelimage1 = IntImageType::New();
  labelimage1->SetSpacing(scalarReader->GetOutput()->GetSpacing());
  labelimage1->SetOrigin(scalarReader->GetOutput()->GetOrigin());
  labelimage1->SetDirection(scalarReader->GetOutput()->GetDirection());
  labelimage1->SetRegions(scalarReader->GetOutput()->GetLargestPossibleRegion());
  labelimage1->Allocate();
  labelimage1->FillBuffer(0);
  labelimageCountF = IntImageType::New();
  labelimageCountF->SetSpacing(scalarReader->GetOutput()->GetSpacing());
  labelimageCountF->SetOrigin(scalarReader->GetOutput()->GetOrigin());
  labelimageCountF->SetDirection(scalarReader->GetOutput()->GetDirection());
  labelimageCountF->SetRegions(scalarReader->GetOutput()->GetLargestPossibleRegion());
  labelimageCountF->Allocate();
  labelimageCountF->FillBuffer(0);
  labelimage2 = IntImageType::New();
  labelimage2->SetSpacing(scalarReader->GetOutput()->GetSpacing());
  labelimage2->SetOrigin(scalarReader->GetOutput()->GetOrigin());
  labelimage2->SetDirection(scalarReader->GetOutput()->GetDirection());
  labelimage2->SetRegions(scalarReader->GetOutput()->GetLargestPossibleRegion());
  labelimage2->Allocate();
  labelimage2->FillBuffer(0);
  labelimage2CountF = IntImageType::New();
  labelimage2CountF->SetSpacing(scalarReader->GetOutput()->GetSpacing());
  labelimage2CountF->SetOrigin(scalarReader->GetOutput()->GetOrigin());
  labelimage2CountF->SetDirection(scalarReader->GetOutput()->GetDirection());
  labelimage2CountF->SetRegions(scalarReader->GetOutput()->GetLargestPossibleRegion());
  labelimage2CountF->Allocate();
  labelimage2CountF->FillBuffer(0);

  
  typedef itk::Point<double, 3> PointType;
  PointType fiberpoint;
  double fiberpointtemp[3];
  for(int i=0;i<SourcePoints->GetNumberOfPoints();i++)
  {	
      SourcePoints->GetPoint( i, fiberpointtemp );
      for (int k=0;k<3;k++)
	{
		fiberpoint[k]=fiberpointtemp[k];
	}
      itk::ContinuousIndex<double,3> cind;
      itk::Index<3> ind;
      labelimage1->TransformPhysicalPointToContinuousIndex(fiberpoint, cind);
      ind[0] = static_cast<long int>(vnl_math_rnd_halfinttoeven(cind[0]));
      ind[1] = static_cast<long int>(vnl_math_rnd_halfinttoeven(cind[1]));
      ind[2] = static_cast<long int>(vnl_math_rnd_halfinttoeven(cind[2]));
	
      if(!labelimage1->GetLargestPossibleRegion().IsInside(ind))
      {
        std::cerr << "Error index: " << ind << " not in image"  << std::endl;
        std::cout << "Ignoring" << std::endl;
        //return EXIT_FAILURE;
      }
      else
      {
	labelimageCountF->SetPixel(ind, labelimageCountF->GetPixel(ind) + 1);
	//std::cout<<"value of pixel: "<<labelimageCountF->GetPixel(ind)<<std::endl;
        labelimage1->SetPixel(ind, voxelLabel);
      }
    }
  PointType fiberpoint2;
  double fiberpointtemp2[3];
  for(int j=0;j<TargetPoints->GetNumberOfPoints();j++)
  {	
      TargetPoints->GetPoint( j, fiberpointtemp2 );
	 for (int k=0;k<3;k++)
	{
		fiberpoint2[k]=fiberpointtemp2[k];
	}
      itk::ContinuousIndex<double,3> cind2;
      itk::Index<3> ind2;
      labelimage2->TransformPhysicalPointToContinuousIndex(fiberpoint2, cind2);
      ind2[0] = static_cast<long int>(vnl_math_rnd_halfinttoeven(cind2[0]));
      ind2[1] = static_cast<long int>(vnl_math_rnd_halfinttoeven(cind2[1]));
      ind2[2] = static_cast<long int>(vnl_math_rnd_halfinttoeven(cind2[2]));
	
      if(!labelimage2->GetLargestPossibleRegion().IsInside(ind2))
      {
        std::cerr << "Error index: " << ind2 << " not in image"  << std::endl;
        std::cout << "Ignoring" << std::endl;
        //return EXIT_FAILURE;
      }
      else
      {
	labelimage2CountF->SetPixel(ind2, labelimage2CountF->GetPixel(ind2) + 1);
        labelimage2->SetPixel(ind2, voxelLabel);
      }
  }

////overlap calculation////
//With classic voxelisation
 typedef itk::LabelOverlapMeasuresImageFilter <IntImageType> LabelOverlapMeasuresImageFilterType;
 LabelOverlapMeasuresImageFilterType::Pointer  LabelOverlapMeasuresImageFilter= LabelOverlapMeasuresImageFilterType::New();
 LabelOverlapMeasuresImageFilter->SetSourceImage(labelimage1);
 LabelOverlapMeasuresImageFilter->SetTargetImage(labelimage2); 
 LabelOverlapMeasuresImageFilter->Update();

double total=LabelOverlapMeasuresImageFilter->GetTotalOverlap();
double jaccard=LabelOverlapMeasuresImageFilter->GetUnionOverlap();
double dice=LabelOverlapMeasuresImageFilter->GetMeanOverlap();
double Volume_sim=LabelOverlapMeasuresImageFilter->GetVolumeSimilarity();
double False_negative=LabelOverlapMeasuresImageFilter->GetFalseNegativeError();
double False_positive=LabelOverlapMeasuresImageFilter->GetFalsePositiveError();
OverlapTable.push_back(total);
OverlapTable.push_back(jaccard);
OverlapTable.push_back(dice);
OverlapTable.push_back(Volume_sim);
OverlapTable.push_back(False_negative);
OverlapTable.push_back(False_positive);
std::cout<<"total: "<<total<<" jaccard: "<<jaccard<<"dice: "<<dice<<"Volume_sim: "<<Volume_sim<<"False_negative: "<<False_negative<<"False_positive: "<<False_positive<<std::endl;
//divide each voxel of Source voxelized by the total number of points to get the probability

IntImageType::Pointer img1 = IntImageType::New();
img1 = labelimageCountF;
itk::ImageRegionIterator<IntImageType> img_it1 (img1, img1->GetLargestPossibleRegion());
std::cout<<" get proba of Source "<<std::endl;
img_it1.GoToBegin();
//divide each voxel of Target voxelized by the total number of points to get the probability
IntImageType::Pointer img2 = IntImageType::New();
img2 = labelimage2CountF;
itk::ImageRegionIterator<IntImageType> img_it2 (img2, img2->GetLargestPossibleRegion());
std::cout<<" get proba of Target "<<std::endl;
img_it2.GoToBegin();
 double numerator=0;
 double denominator=0;

while(!img_it1.IsAtEnd() && !img_it2.IsAtEnd())
    {
    // Get the value of the current voxel
     double val1 = img_it1.Get(); 
     double  val2= img_it2.Get();
     //Get the proba on each voxel
     double Pa=val1/numberPointsSource;
     double Pb=val2/numberPointsTarget;
     numerator+=abs(Pa-Pb);
     denominator+=(Pa+Pb-(Pa*Pb));
     ++img_it1;
     ++img_it2;
    }

std::cout<<"points in source : "<<numberPointsSource<<std::endl;
std::cout<<"points in target : "<<numberPointsTarget<<std::endl;

std::cout<<"compute POV  "<<std::endl;
//Compute POV calculation
double POV=1-(numerator/denominator);
std::cout<<"POV : "<< POV<<std::endl;
	
OverlapTable.push_back(POV);
return OverlapTable;
}




std::vector<double> FillCurvatureTable(vtkSmartPointer<vtkPolyData> SourceInterpolated, vtkSmartPointer<vtkPolyData> TargetInterpolated, double stepInterpolate)
{	std::cout<<"Curvature metric calculating..."<<std::endl;
	std::vector<double> CurveTable;
	vtkPoints* SourcePointsNew = SourceInterpolated->GetPoints();
	vtkPoints* TargetPointsNew = TargetInterpolated->GetPoints();
	typedef itk::Vector<double, 3> VectorType;
	std::cout<<" number of points in source: "<<SourcePointsNew->GetNumberOfPoints()<<std::endl;
	std::cout<<" number of points in target: "<<TargetPointsNew->GetNumberOfPoints()<<std::endl;

	vtkIdType nbpts=0, *pts=0;
	double Tbis[3],Tbis2[3];
	double T[3],T2[3];
	double curvature1=0;
	double curvature2=0;
	double dsbis=0;
	double dsbis2=0;
	double ds=0;
	double ds2=0;
	int compt=1;
	int compt2=1;
	vtkSmartPointer<vtkCellArray> A = vtkSmartPointer<vtkCellArray>::New();
	A=SourceInterpolated->GetLines();
	A->InitTraversal();
	std::cout<<" number of fibers in source: "<<SourceInterpolated->GetNumberOfLines()<<std::endl;
	while(SourceInterpolated->GetLines()->GetNextCell(nbpts, pts))
	{	std::cout<<" fibers in source number: "<<compt2<<std::endl;
		vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
 		vtkIdType ind=0;double coords[3];
        	for(int p=0; p<nbpts; p++)
        	{
            		ind=pts[p];
			SourcePointsNew->GetPoint( ind, coords);
			points->InsertNextPoint(coords);
		}
		//for each point of hte current fiber with its new points from the spline interpolation	
		int i=0;
    		for( int k=0; k<points->GetNumberOfPoints(); k++ )
		{	if(k==(points->GetNumberOfPoints()-1))
			{
				i=k-1;
			}
			else
			{
				i=k;
			}
			double DistanceMin=100000;
			double SourceP[3];
			points->GetPoint(i,SourceP);
			double TargetPoint[3];
			double Distance=0;
			////for each fiber of vtk file Target////
			vtkIdType nbpts2=0, *pts2=0;
			vtkSmartPointer<vtkCellArray> P = vtkSmartPointer<vtkCellArray>::New();
			P=TargetInterpolated->GetLines();
			P->InitTraversal();
  			while(TargetInterpolated->GetLines()->GetNextCell(nbpts2, pts2))
			{	
				vtkSmartPointer<vtkPoints> points2 = vtkSmartPointer<vtkPoints>::New();
 				vtkIdType ind2=0;double coords2[3];
        			for(int p=0; p<nbpts2; p++)
        			{
            				ind2=pts2[p];
					TargetPointsNew->GetPoint( ind2, coords2);
					points2->InsertNextPoint(coords2);
				}
				//for each point of the current fiber with its new points from the spline interpolation	
				int j=0;
      				for( int l=0; l<points2->GetNumberOfPoints(); l++ )
				{	
					if(l==(points2->GetNumberOfPoints()-1))
					{
						j=l-1;
					}
					else
					{
						j=l;
					}
					points2->GetPoint(j,TargetPoint);
					
					Distance=vtkMath::Distance2BetweenPoints(SourceP,TargetPoint );
					if(Distance<DistanceMin )
					{	
						//std::cout<<"two closest points"<<SourceP[0]<<" "<<SourceP[1]<<" "<<SourceP[2]<<"and "<<TargetPoint[0]<<" "<<TargetPoint[1]<<" "<<TargetPoint[2]<<std::endl;
	  					DistanceMin = Distance;//std::cout<<" distance : "<<DistanceMin<<std::endl;
	  					double matchingpoint[3];matchingpoint[0]=TargetPoint[0];matchingpoint[1]=TargetPoint[1];matchingpoint[2]=TargetPoint[2];
						double tempo2[3],tempo22[3];
						points2->GetPoint(j+1,tempo2);
						points2->GetPoint(j+2,tempo22);
						ds = sqrt(vtkMath::Distance2BetweenPoints(TargetPoint,tempo2));
						ds2= sqrt(vtkMath::Distance2BetweenPoints(tempo2,tempo22));
						T[0] = (tempo2[0]-TargetPoint[0])/ds;
						T[1] = (tempo2[1]-TargetPoint[1])/ds;
						T[2] = (tempo2[2]-TargetPoint[2])/ds;
						T2[0] = (tempo22[0]-tempo2[0])/ds2;
						T2[1] = (tempo22[1]-tempo2[1])/ds2;
						T2[2] = (tempo22[2]-tempo2[2])/ds2;
						curvature2=sqrt(vtkMath::Distance2BetweenPoints(T2,T))/ds;

					}
      				}
				
    			}
			
			double temp2[3],temp22[3];
			points->GetPoint(i+1,temp2);
			points->GetPoint(i+2,temp22);
			dsbis = sqrt(vtkMath::Distance2BetweenPoints(SourceP,temp2));
			dsbis2= sqrt(vtkMath::Distance2BetweenPoints(temp2,temp22));
			Tbis[0] = (temp2[0]-SourceP[0])/dsbis;
			Tbis[1] = (temp2[1]-SourceP[1])/dsbis;
			Tbis[2] = (temp2[2]-SourceP[2])/dsbis;
			Tbis2[0] = (temp22[0]-temp2[0])/dsbis2;
			Tbis2[1] = (temp22[1]-temp2[1])/dsbis2;
			Tbis2[2] = (temp22[2]-temp2[2])/dsbis2;
			curvature1=sqrt(vtkMath::Distance2BetweenPoints(Tbis2,Tbis))/dsbis;

			double val =(curvature1-curvature2)*(curvature1-curvature2);
			//std::cout<<"   val : "<<val<<" number : "<<compt<<std::endl;
			compt++;
			CurveTable.push_back(val);
			
		}
	compt2++;	

	}
	std::cout<<"Done..."<<std::endl;
	return CurveTable;

	
}


std::vector<double> FillTangentTable(vtkSmartPointer<vtkPolyData> SourceInterpolated, vtkSmartPointer<vtkPolyData> TargetInterpolated, double stepInterpolate)
{	std::cout<<"Tangent metric calculating..."<<std::endl;
	std::vector<double> TangentTable;
	std::vector<double> samples1, samples2;
	//for each point of the current fiber with its new points from the spline interpolation	
	vtkPoints* SourcePointsNew = SourceInterpolated->GetPoints();
	vtkPoints* TargetPointsNew = TargetInterpolated->GetPoints();
	
	vtkIdType nbpts=0, *pts=0;
	typedef itk::Vector<double, 3> VectorType;
	VectorType v1, v2;		
	std::cout<<" number of points in source: "<<SourcePointsNew->GetNumberOfPoints()<<std::endl;
	std::cout<<" number of points in target: "<<TargetPointsNew->GetNumberOfPoints()<<std::endl;

	int compt=1;
	std::cout<<" number of fibers in source: "<<SourceInterpolated->GetNumberOfLines()<<std::endl;
	vtkSmartPointer<vtkCellArray> A = vtkSmartPointer<vtkCellArray>::New();
	A=SourceInterpolated->GetLines();
	A->InitTraversal();
	while(SourceInterpolated->GetLines()->GetNextCell(nbpts, pts))
	{	//std::cout<<" fibers in source number: "<<compt<<std::endl;
		vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
 		vtkIdType ind=0;double coords[3];
        	for(int p=0; p<nbpts; p++)
        	{
            		ind=pts[p];
			SourcePointsNew->GetPoint( ind, coords);
			points->InsertNextPoint(coords);
		}
		
		//std::cout<<"number of points: "<<points->GetNumberOfPoints()<<std::endl;
    		for( int i=0; i<points->GetNumberOfPoints(); i++ )
		{
			double DistanceMin=100000;
			double SourceP[3];
			points->GetPoint(i,SourceP);	
			double TargetPoint[3];
			double Distance=0;
			vtkIdType nbpts2=0, *pts2=0;
			vtkSmartPointer<vtkCellArray> P = vtkSmartPointer<vtkCellArray>::New();
			P=TargetInterpolated->GetLines();
			P->InitTraversal();
			while(TargetInterpolated->GetLines()->GetNextCell(nbpts2, pts2))
			{
				
				vtkSmartPointer<vtkPoints> points2 = vtkSmartPointer<vtkPoints>::New();
 				vtkIdType ind2=0;double coords2[3];
				
        			for(int p=0; p<nbpts2; p++)
        			{
            				ind2=pts2[p];
					TargetPointsNew->GetPoint( ind2, coords2);
					points2->InsertNextPoint(coords2);
				}
			//for each point of the current fiber with its new points from the spline interpolation	
      				for( int j=0; j<points2->GetNumberOfPoints(); j++ )
				{	
					points2->GetPoint(j,TargetPoint);
					
					Distance=vtkMath::Distance2BetweenPoints(SourceP,TargetPoint );
					if(Distance<DistanceMin )
					{
	  					DistanceMin = Distance;
						//std::cout<<"distance min :"<<DistanceMin<<std::endl;
	  					double matchingpoint[3];matchingpoint[0]=TargetPoint[0];matchingpoint[1]=TargetPoint[1];matchingpoint[2]=TargetPoint[2];
						double tempo1[3],tempo2[3];
						
						if (j==0)////if we are at the beginning of the fiber////
						{
					 		points2->GetPoint(j+1,tempo2);
							for(int k=0;k<3;k++)
							{
								v2[k]=tempo2[k]-TargetPoint[k];
							}
						}
						if(j==points2->GetNumberOfPoints()-1)////if we are at the end of the fiber////
						{	
							points2->GetPoint(j-1,tempo1);
							for(int k=0;k<3;k++)
							{
								v2[k]=TargetPoint[k]-tempo1[k];
							}

						}
						if(j!=0 && j!=points2->GetNumberOfPoints()-1)
						{	
							points2->GetPoint(j-1,tempo1);
							points2->GetPoint(j+1,tempo2);
							for(int k=0;k<3;k++)
							{
								v2[k]=tempo2[k]-tempo1[k];
							}

						}

					}
      				}
		
			}	
    	
		double temp1[3],temp2[3];
		//std::cout<<"i : "<<i<<std::endl;
		if (i==0)////if we are at the beginning of the fiber////
		{ 
			points->GetPoint(i+1,temp2);
			for(int k=0;k<3;k++)
			{
				v1[k]=temp2[k]-SourceP[k];
			}
		}
		if (i==points->GetNumberOfPoints()-1)////if we are at the end of the fiber////
		{
			points->GetPoint(i-1,temp1);
			for(int k=0;k<3;k++)
			{
				v1[k]=SourceP[k]-temp1[k];
			}
				
		}
		if(i!=0 && i!=points->GetNumberOfPoints()-1)
		{
			points->GetPoint(i-1,temp1);
			points->GetPoint(i+1,temp2);
			for(int k=0;k<3;k++)
			{
				v1[k]=temp2[k]-temp1[k];
			}
				
		}

    		v1.Normalize();
    		v2.Normalize();
		//std::cout<<v1[0]<<" "<<v1[1]<<" "<<v1[2]<<std::endl;
		//std::cout<<v2[0]<<" "<<v2[1]<<" "<<v2[2]<<std::endl;

		double val = pow ( acos ( fabs ( v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]) )*180/3.14159265 , 2.0);
		/*if(compt==211)
			std::cout<<"val : "<<(acos ( fabs ( v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]) )*180/3.14159265)<<std::endl;*/
		TangentTable.push_back(val);
			
		}
	
	compt++;
		
	}
	std::cout<<"done..."<<std::endl;	
	return TangentTable;
}


std::vector<double> FillDistanceTable(vtkSmartPointer<vtkPolyData> Source, vtkSmartPointer<vtkPolyData> Target)
{
	std::cout<<"Distance (closest points) metric calculating..."<<std::endl;
	std::vector<double> DistanceTable;
	vtkPoints* SourcePoints = Source->GetPoints();
	vtkPoints* TargetPoints = Target->GetPoints();
	
	for(int i=0; i<SourcePoints->GetNumberOfPoints(); i++)
	{
		double SourcePoint[3], DistanceMin = 10000;
		SourcePoints->GetPoint(i,SourcePoint);
		for(int j=0; j<TargetPoints->GetNumberOfPoints(); j++)
		{
			double TargetPoint[3], XYZ[3], Distance;
			TargetPoints->GetPoint(j,TargetPoint);
			for(int k=0; k<3; k++)
				XYZ[k] = TargetPoint[k]-SourcePoint[k];
			Distance = sqrt(XYZ[0]*XYZ[0]+XYZ[1]*XYZ[1]+XYZ[2]*XYZ[2]);
			if(Distance<DistanceMin)
				DistanceMin = Distance;
		}
		DistanceTable.push_back(DistanceMin);
	}
	std::cout<<"Done..."<<std::endl;
	return DistanceTable;
}

void GetBounds(std::vector<std::vector<double> > DistanceTable, double Bounds[2])
{
	double Min = 100000, Max = -1;
	for(unsigned int i=0; i<DistanceTable.size(); i++)
	{
		for(unsigned int j=0; j<DistanceTable[i].size(); j++)
		{
			if(DistanceTable[i][j]<Min)
				Min = DistanceTable[i][j];
			if(DistanceTable[i][j]>Max)
				Max = DistanceTable[i][j];
		}
	}
	Bounds[0] = Min;
	Bounds[1] = Max;
}

int GetFrequency(std::vector<std::vector<double> > DistanceTable, double IntervalMin, double IntervalMax)
{
	int Frequency=0;
	for(unsigned int i=0; i<DistanceTable.size(); i++)
	{
		for(unsigned int j=0; j<DistanceTable[i].size(); j++)
		{
			if(DistanceTable[i][j]<IntervalMax && DistanceTable[i][j]>=IntervalMin)
				Frequency++;
		}
	}
	return Frequency;
}

double GetMeanDistance(std::vector<double> Distance, std::vector<double> Frequency)
{
	double MeanDistance=0, TotalFrequency=0;
	for(unsigned int i=0; i<Distance.size(); i++)
	{
		MeanDistance+=Distance[i]*Frequency[i];
		TotalFrequency+=Frequency[i];
	}
	MeanDistance/=TotalFrequency;
	
	return MeanDistance;
}


std::vector<std::vector<double> > GetResultTableFrequency(std::vector<std::vector<double> >  Table,int TotalNumberOfFibers,int number_of_entries,double step)
{
	double Bounds[2];
	GetBounds(Table,Bounds);
	if(step==-1)
		step=(Bounds[1]-Bounds[0])/number_of_entries;
	std::cout<<Bounds[0]<<" "<<Bounds[1]<<" "<<number_of_entries<<" "<<step<<std::endl;
	double IntervalMin = Bounds[0], IntervalMax = Bounds[0] + step;
	std::vector<double> FrequencyVector;
	std::vector<double> DistanceVector;
		
	while(IntervalMin<=Bounds[1])
	{
		int Frequency = GetFrequency(Table, IntervalMin, IntervalMax);
		DistanceVector.push_back(IntervalMin);
		FrequencyVector.push_back(Frequency);
		IntervalMin += step;
		IntervalMax += step;
	}
		
	double Dist25=-1, Dist50=-1, Dist75=-1, Dist90=-1, Dist95=-1;
	double CumulatedFrequency=0;

	for(unsigned int i=0; i<DistanceVector.size(); i++)
	{
		//StatFile<<DistanceVector[i]<<","<<FrequencyVector[i]<<std::endl;
		CumulatedFrequency+=FrequencyVector[i];
		//std::cout<<"cumulated frequency"<<CumulatedFrequency<<"TotalNumberOfFibers"<<TotalNumberOfFibers<<std::endl;
		if(CumulatedFrequency>=TotalNumberOfFibers*0.25 && Dist25==-1)
		{
			Dist25=DistanceVector[i];
		}
		if(CumulatedFrequency>=TotalNumberOfFibers*0.5 && Dist50==-1)
		{
			Dist50=DistanceVector[i];
		}
		if(CumulatedFrequency>=TotalNumberOfFibers*0.75 && Dist75==-1)
		{
			Dist75=DistanceVector[i];
		}
		if(CumulatedFrequency>=TotalNumberOfFibers*0.9 && Dist90==-1)
		{
			Dist90=DistanceVector[i];
		}
		if(CumulatedFrequency>=TotalNumberOfFibers*0.95 && Dist95==-1)
		{
			Dist95=DistanceVector[i];
		}
	}
	std::vector<std::vector<double> > ResultDistanceFrequency;
	std::vector<double> DistanceFrequency;
	DistanceFrequency.push_back(Dist25);
	DistanceFrequency.push_back(Dist50);
	DistanceFrequency.push_back(Dist75);
	DistanceFrequency.push_back(Dist90);
	DistanceFrequency.push_back(Dist95);
	DistanceFrequency.push_back(Bounds[0]);
	DistanceFrequency.push_back(Bounds[1]);
	ResultDistanceFrequency.push_back(DistanceFrequency);
	ResultDistanceFrequency.push_back(DistanceVector);
	ResultDistanceFrequency.push_back(FrequencyVector);
	

 return ResultDistanceFrequency;

}




int main(int argc, char* argv[])
{
	PARSE_ARGS;

	int Number_of_entries=number_of_entries;
	double Step=step;
	std::vector<vtkSmartPointer<vtkPolyData> > FiberTracts;
	std::vector<std::string> Filenames;

	std::cout<<"Reading VTK data..."<<std::endl;
	
	Filenames.push_back(vtk_input1);
	Filenames.push_back(vtk_input2);
	for(int i=0; i<2; i++)
	{
		if(Filenames[i].rfind(".vtk") != std::string::npos)
		{
			vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
			reader->SetFileName(Filenames[i].c_str());
			FiberTracts.push_back(reader->GetOutput());
			reader->Update();
		}
		else if(Filenames[i].rfind(".vtp") != std::string::npos)
		{
			vtkSmartPointer<vtkXMLPolyDataReader> reader = vtkSmartPointer<vtkXMLPolyDataReader>::New();
			reader->SetFileName(Filenames[i].c_str());
			FiberTracts.push_back(reader->GetOutput());
			reader->Update();
		}
		else
		{
			std::cout<<"Wrong filename : "<<Filenames[i]<<" Check file format and location."<<std::endl;
			return 0;
		}
	}
	std::cout<<Filenames[0]<<" "<<FiberTracts[0]->GetNumberOfCells()<<" fibers, "<<FiberTracts[0]->GetPoints()->GetNumberOfPoints()<<" points."<<std::endl;
	std::cout<<Filenames[1]<<" "<<FiberTracts[1]->GetNumberOfCells()<<" fibers, "<<FiberTracts[1]->GetPoints()->GetNumberOfPoints()<<" points."<<std::endl;
	std::cout<<"VTK Files read successfuly."<<std::endl<<std::endl;

	std::cout<<"interpolation"<<std::endl<<std::endl;
	vtkSmartPointer<vtkPolyData> Source = vtkSmartPointer<vtkPolyData>::New();
	Source=FiberTracts[0];
	vtkSmartPointer<vtkPolyData> Target= vtkSmartPointer<vtkPolyData>::New();
	Target=FiberTracts[1];
	vtkPoints* SourcePoints = Source->GetPoints();
	vtkPoints* TargetPoints = Target->GetPoints();
	std::cout<<"points "<<SourcePoints->GetNumberOfPoints()<<std::endl;
	std::cout<<"points "<<TargetPoints->GetNumberOfPoints()<<std::endl;
	//new vtk with spline interpolation
	vtkSmartPointer<vtkPolyData> SourceInterpolated = vtkSmartPointer<vtkPolyData>::New();
	vtkSmartPointer<vtkPolyData> TargetInterpolated = vtkSmartPointer<vtkPolyData>::New();
	typedef itk::Vector<double, 3> VectorType;
	
	vtkIdType nbpts=0, *pts=0;
	VectorType v1, v2;
	////for each fiber of vtk file Source////
	vtkSmartPointer <vtkCellArray> cells = vtkSmartPointer <vtkCellArray>::New();
	int countID=0;
	vtkPoints* polypoints = vtkPoints::New();
  	while(Source->GetLines()->GetNextCell(nbpts, pts))
	{	
		//store points of the current fiber in points
		vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
 		vtkIdType ind=0;double coords[3];
        	for(int p=0; p<nbpts; p++)
        	{
            		ind=pts[p];
			SourcePoints->GetPoint( ind, coords);
			points->InsertNextPoint(coords);
		}
		//spline interpolation
		int numberOfInputPoints = points->GetNumberOfPoints();
   		
    		vtkCardinalSpline* aSplineX;
    		vtkCardinalSpline* aSplineY;
   		vtkCardinalSpline* aSplineZ;    
   		aSplineX = vtkCardinalSpline::New();
   		aSplineY = vtkCardinalSpline::New();
    		aSplineZ = vtkCardinalSpline::New();
		aSplineX->ClosedOff();
		aSplineY->ClosedOff();
		aSplineZ->ClosedOff();

		/////creation of splines/////
   		 for (int i=0; i<numberOfInputPoints; i++) 
   		 {
		  
    		  double x = points->GetPoint(i)[0];
    		  double y = points->GetPoint(i)[1];
     		  double z = points->GetPoint(i)[2];
          	  aSplineX->AddPoint(i, x);
          	  aSplineY->AddPoint(i, y);
          	  aSplineZ->AddPoint(i, z);
    		 }
	

		double  tRange[2];
		aSplineX->GetParametricRange 	( tRange); 
		int compteur2=0;
    		
		////evaluation of new points then store it in polypoints////
		for( double t = 0 ; t <= tRange[1]; t += stepInterpolate )
    		{
      		polypoints->InsertNextPoint(aSplineX->Evaluate(t),aSplineY->Evaluate(t),aSplineZ->Evaluate(t));
		//compteur2 gives us the number of new points of the current fiber
		compteur2++;
    		}
		//creation of polylines
		vtkSmartPointer<vtkPolyLine> polyline = vtkSmartPointer<vtkPolyLine>::New();
		polyline->GetPointIds()->SetNumberOfIds(compteur2);
		//countID gives us the total number of points cumulated
		countID=polypoints->GetNumberOfPoints();
		
		for (int i= 0; i<compteur2 ;i++)
		{	//SETID(0->number of points in the current fiber, (total points cumulated-points in the current fiber)->total points cumulated)
			polyline->GetPointIds()->SetId(i,i+(countID-compteur2));
		}
		cells->InsertNextCell(polyline);
		aSplineX->Delete();
		aSplineY->Delete();
		aSplineZ->Delete();
		
	}
	SourceInterpolated->SetPoints(polypoints);
	SourceInterpolated->SetLines(cells);
	polypoints->Delete();
	std::cout<<"end of the first interpolation"<<std::endl<<std::endl;

	vtkIdType nbpts2=0, *pts2=0;
	vtkSmartPointer <vtkCellArray> cells2 = vtkSmartPointer <vtkCellArray>::New();
	int countID2=0;
	vtkPoints* polypoints2 = vtkPoints::New();
	while(Target->GetLines()->GetNextCell(nbpts2, pts2))
	{	
		//std::cout<<"step2"<<std::endl;
		vtkSmartPointer<vtkPoints> points2 = vtkSmartPointer<vtkPoints>::New();
 		vtkIdType ind2=0;double coords2[3];
        	for(int p2=0; p2<nbpts2; p2++)
        	{
            		ind2=pts2[p2];
			TargetPoints->GetPoint( ind2, coords2);
			points2->InsertNextPoint(coords2);
		}
		//spline interpolation
		int numberOfInputPoints2 = points2->GetNumberOfPoints();
    		vtkCardinalSpline* aSpline2X;
    		vtkCardinalSpline* aSpline2Y;
   		vtkCardinalSpline* aSpline2Z;    

   		aSpline2X = vtkCardinalSpline::New();
   		aSpline2Y = vtkCardinalSpline::New();
    		aSpline2Z = vtkCardinalSpline::New();
		aSpline2X->ClosedOff();
		aSpline2Y->ClosedOff();
		aSpline2Z->ClosedOff();
		/////creation of splines/////
   		 for (int i=0; i<numberOfInputPoints2; i++) 
   		{
			//std::cout<<"step6"<<std::endl;
    		 	double x2 = points2->GetPoint(i)[0];
    		 	double y2 = points2->GetPoint(i)[1];
     		 	double z2 = points2->GetPoint(i)[2];
          	 	aSpline2X->AddPoint(i, x2);
          	  	aSpline2Y->AddPoint(i, y2);
          	  	aSpline2Z->AddPoint(i, z2);
    		}
		double  tRange2[2];
		aSpline2X->GetParametricRange 	( tRange2);
    		int compteur=0;
		////evaluation of new points then store it in polypoints2////
		for( double t2 = 0 ; t2 <= tRange2[1]; t2 += stepInterpolate )
    		{	
      		polypoints2->InsertNextPoint(aSpline2X->Evaluate(t2),aSpline2Y->Evaluate(t2),aSpline2Z->Evaluate(t2));
		compteur++;
    		}
		vtkSmartPointer<vtkPolyLine> polyline2 = vtkSmartPointer<vtkPolyLine>::New();
		polyline2->GetPointIds()->SetNumberOfIds(compteur);
		countID2=polypoints2->GetNumberOfPoints();
		for (int i=0; i<compteur;i++)
		{
			polyline2->GetPointIds()->SetId(i,i+(countID2-compteur));
		}
		cells2->InsertNextCell(polyline2);
		aSpline2X->Delete();
		aSpline2Y->Delete();
		aSpline2Z->Delete();
		
		//end 

	}
	TargetInterpolated->SetPoints(polypoints2);
	TargetInterpolated->SetLines(cells2);
	polypoints2->Delete();
	std::cout<<"end of the second interpolation"<<std::endl<<std::endl;

	std::cout<<"Calculating..."<<std::endl;
			
	std::vector<std::vector<double> > DistanceTable;
	DistanceTable.push_back(FillDistanceTable(SourceInterpolated,TargetInterpolated));
	DistanceTable.push_back(FillDistanceTable(TargetInterpolated,SourceInterpolated));

	std::vector<std::vector<double> > TangentTable;
	TangentTable.push_back(FillTangentTable(SourceInterpolated,TargetInterpolated,stepInterpolate));
	TangentTable.push_back(FillTangentTable(TargetInterpolated,SourceInterpolated,stepInterpolate));
		
	std::vector<std::vector<double> > CurveTable;
	CurveTable.push_back(FillCurvatureTable(SourceInterpolated,TargetInterpolated,stepInterpolate));
	CurveTable.push_back(FillCurvatureTable(TargetInterpolated,SourceInterpolated,stepInterpolate));
	
	int TotalNumberOfFibers=SourceInterpolated->GetPoints()->GetNumberOfPoints()+TargetInterpolated->GetPoints()->GetNumberOfPoints();
	
		
	std::ofstream StatFile(output_stat_file.c_str(), std::ios::out);
	std::cout<<"number total points"<<TotalNumberOfFibers<<std::endl;	
	if(StatFile)
	{
		StatFile<<"Filename,Number of fibers,Number of points"<<std::endl;
		StatFile<<Filenames[0]<<","<<SourceInterpolated->GetNumberOfCells()<<","<<SourceInterpolated->GetPoints()->GetNumberOfPoints()<<std::endl;
		StatFile<<Filenames[1]<<","<<TargetInterpolated->GetNumberOfCells()<<","<<TargetInterpolated->GetPoints()->GetNumberOfPoints()<<std::endl<<std::endl;
		std::vector<std::vector<double> > DistanceTableFrequency=GetResultTableFrequency(DistanceTable,TotalNumberOfFibers,Number_of_entries,Step);
		std::cout<<"frequency tangent metric"<<std::endl;
		std::vector<std::vector<double> > TangentTableFrequency=GetResultTableFrequency(TangentTable,TotalNumberOfFibers,Number_of_entries,Step);
		std::cout<<"frequency curve metric"<<std::endl;
		std::vector<std::vector<double> > CurveTableFrequency=GetResultTableFrequency(CurveTable,TotalNumberOfFibers,Number_of_entries,Step);

		StatFile<<"Distance,Frequency closest points"<<std::endl;
		std::vector<double> FrequencyVector1=DistanceTableFrequency[2];
		std::vector<double> DistanceVector1=DistanceTableFrequency[1];
		for(unsigned int i=0; i<DistanceVector1.size(); i++)
		{
			StatFile<<DistanceVector1[i]<<","<<FrequencyVector1[i]<<std::endl;
		}
		StatFile<<std::endl;

		StatFile<<"Distance,Frequency of tangent metric"<<std::endl;
		std::vector<double> FrequencyVector2=TangentTableFrequency[2];
		std::vector<double> DistanceVector2=TangentTableFrequency[1];
		for(unsigned int i=0; i<DistanceVector2.size(); i++)
		{
			StatFile<<DistanceVector2[i]<<","<<FrequencyVector2[i]<<std::endl;
		}
		StatFile<<std::endl;

		StatFile<<"Distance,Frequency of curve metric"<<std::endl;
		std::vector<double> FrequencyVector3=CurveTableFrequency[2];
		std::vector<double> DistanceVector3=CurveTableFrequency[1];
		for(unsigned int i=0; i<DistanceVector3.size(); i++)
		{
			StatFile<<DistanceVector3[i]<<","<<FrequencyVector3[i]<<std::endl;
		}
		StatFile<<std::endl;

		StatFile<<"Distance Frequency Percentage,Distance"<<std::endl;
		StatFile<<"25%,"<<DistanceTableFrequency[0][0]<<std::endl;
		StatFile<<"50%,"<<DistanceTableFrequency[0][1]<<std::endl;
		StatFile<<"75%,"<<DistanceTableFrequency[0][2]<<std::endl;
		StatFile<<"90%,"<<DistanceTableFrequency[0][3]<<std::endl;
		StatFile<<"95%,"<<DistanceTableFrequency[0][4]<<std::endl;
		StatFile<<std::endl;
		StatFile<<"Tangent metric Frequency Percentage,Distance"<<std::endl;
		StatFile<<"25%,"<<TangentTableFrequency[0][0]<<std::endl;
		StatFile<<"50%,"<<TangentTableFrequency[0][1]<<std::endl;
		StatFile<<"75%,"<<TangentTableFrequency[0][2]<<std::endl;
		StatFile<<"90%,"<<TangentTableFrequency[0][3]<<std::endl;
		StatFile<<"95%,"<<TangentTableFrequency[0][4]<<std::endl;
		StatFile<<std::endl;
		StatFile<<"Curvature metric Frequency Percentage,Distance"<<std::endl;
		StatFile<<"25%,"<<CurveTableFrequency[0][0]<<std::endl;
		StatFile<<"50%,"<<CurveTableFrequency[0][1]<<std::endl;
		StatFile<<"75%,"<<CurveTableFrequency[0][2]<<std::endl;
		StatFile<<"90%,"<<CurveTableFrequency[0][3]<<std::endl;
		StatFile<<"95%,"<<CurveTableFrequency[0][4]<<std::endl;
		StatFile<<std::endl;
		

		for(unsigned int i=0; i<methods.size(); i++)
		{	
			if(methods[i] == "Overlap")
			{	if( ReferenceScalarVolume == "")
    				{
      					std::cerr << "A reference volume has to be specified" << std::endl;
     					 return EXIT_FAILURE;
   				 }
				std::vector<double> OverlapTable;
				OverlapTable=FillOverlapTable(FiberTracts[0],FiberTracts[1],ReferenceScalarVolume,voxelLabel);
				StatFile<<"Total volumetric overlap, "<<OverlapTable[0]<<std::endl;
				StatFile<<"Union volumetric overlap (Jaccard coefficient), "<<OverlapTable[1]<<std::endl;
				StatFile<<"Mean volumetric overlap (Dice coefficient), "<<OverlapTable[2]<<std::endl;
				StatFile<<"Volume similarity, "<<OverlapTable[3]<<std::endl;
				StatFile<<"False negative error, "<<OverlapTable[4]<<std::endl;
				StatFile<<"False positive error, "<<OverlapTable[5]<<std::endl;
				StatFile<<"Probabilistic overlap (POV), "<<OverlapTable[6]<<std::endl;
			}

			if(methods[i] == "Hausdorff")
			{
				std::cout<<"	Hausdorff..."<<std::endl;
				
				StatFile<<"Distance: 100%,"<<DistanceTableFrequency[0][6]<<std::endl;
				StatFile<<"Tangent metric: 100%,"<<TangentTableFrequency[0][6]<<std::endl;
				StatFile<<"Curve metric: 100%,"<<CurveTableFrequency[0][6]<<std::endl;
				
				std::cout<<"	End of Hausdorff."<<std::endl;
			}
			if(methods[i] == "Mean")
			{
				std::cout<<"	Mean..."<<std::endl;
				
				StatFile<<"Distance Mean,"<<GetMeanDistance(DistanceVector1,FrequencyVector1)<<std::endl;
				StatFile<<"Tangent metric  Mean,"<<GetMeanDistance(DistanceVector2,FrequencyVector2)<<std::endl;
				StatFile<<"Curve metric Mean,"<<GetMeanDistance(DistanceVector3,FrequencyVector3)<<std::endl;
				
				std::cout<<"	End of Mean."<<std::endl;
			}
			else if(methods[i] == "None")
			{
				std::cout<<"Ignoring Mean and Hausdorff methods"<<std::endl;
			}
			/*else
			{
				std::cout<<"Wrong method or syntax for argument, none option is selected : "<<methods[i]<<std::endl;
				std::cout<<"Ignoring argument."<<std::endl;
			}*/
		}
	StatFile.close();
	}
	else
		std::cout<<"ERROR: Unable to save output stat file."<<std::endl;
	
	std::cout<<"Calculation complete."<<std::endl;
	
	return 0;
}

