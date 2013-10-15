#include <iostream>
#include <fstream>
#include <boost/lexical_cast.hpp>

#include <cstring>

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glut.h>
#include <GL/glext.h>

GLuint _texture;
float colors[3*256];


void display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
    glEnable( GL_TEXTURE_2D );

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f,0.0f); glVertex2f(-0.5f,-0.5f);
	glTexCoord2f(1.0f,0.0f); glVertex2f( 0.5f,-0.5f);
	glTexCoord2f(1.0f,1.0f); glVertex2f( 0.5f, 0.5f);
	glTexCoord2f(0.0f,1.0f); glVertex2f(-0.5f, 0.5f);
	glEnd();  

	glutSwapBuffers();
}

int main(int argc, char** argv)
{

	if (argc == 1)
	{
		std::cerr<<"Try 'transferFunctionCreator --help' for more information."<<std::endl;
		return 0;
	}

	if (argc == 2)
	{
		if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)
		{
			std::cerr<<"Allow to create a transfer function color file"<<std::endl;
			std::cerr<<"At least, three colors have to be provided"<<std::endl;
			std::cerr<<"Usage: transferFunctionCreator color_1 color_2 [color_3 .. color_n] color_background output_file"<<std::endl;
			std::cerr<<"Each color are defined by three integers [0, 255]"<<std::endl;
			std::cerr<<"Example transferFunctionCreator 234 32 0 200 140 42 255 255 255"<<std::endl;
			return 0;
		}
		else
		{
			std::cerr<<"Try 'transferFunctionCreator --help' for more information."<<std::endl;
			return 0;
		}
		
	}

	if (argc < 11 || argc % 3 != 2)
	{
			std::cerr<<"Try 'transferFunctionCreator --help' for more information."<<std::endl;
			return 0;
	}

	int numColor = (argc - 5)/3;
	float baseColors[numColor][3];
	float backColor[3];

	int index = 1;
	for(int i=0; i<numColor + 1; i++)
		for(int j=0; j<3; j++)
		{
			try
			{
				int c  = boost::lexical_cast<int>(argv[index]); 
				if (c < 0 || c > 255)
				{
					std::cerr<<"Error "<<c<<" is out of range [0 ... 255]"<<std::endl;
					return 0;
				}
				if (i < numColor)
					baseColors[i][j] = c/255.0f;
				else
					backColor[j] = c/255.0f;
			}
			catch( ... )
			{
				std::cerr<<"Error "<<argv[index]<<" is not a integer"<<std::endl;
				return 0;
			}
			index++;
		}

	int numInt = 255 / (numColor-1);
	int interval[numColor];
	interval[0] = 0;
	interval[numColor-1] = 255;
	for(int i=1; i<numColor-1; i++)
		interval[i] = interval[i-1]+numInt;

	index=0;
	for(int i=0; i<numColor-1; i++)
	{
		for(int j=interval[i]; j<=interval[i+1]; j++)
		{
			
			colors[index] = baseColors[i][0] + (baseColors[i+1][0] - baseColors[i][0])*(((float)j-(float)interval[i])/(float)(interval[i+1]-interval[i]));
			colors[index+1] = baseColors[i][1] + (baseColors[i+1][1] - baseColors[i][1])*(((float)j-(float)interval[i])/(float)(interval[i+1]-interval[i]));
			colors[index+2] = baseColors[i][2] + (baseColors[i+1][2] - baseColors[i][2])*(((float)j-(float)interval[i])/(float)(interval[i+1]-interval[i]));
			index+=3;
		}
	}

	std::ofstream file(argv[argc-1], std::ofstream::binary);

	for(int i=0; i<3; i++)
	{
		index = 0;
		for(int j=0; j<=256; j++)
		{
			if (j<=255)
				file.write((char*)&colors[index + i],  sizeof(float));
			else
				file.write((char*)&backColor[i],  sizeof(float));

			index+=3;
		}
	}

	file.close();	

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(512, 512);
	glutCreateWindow("Transfer Function Creator");

	glutDisplayFunc(display);

	//CREATE TEXTURE
    glGenTextures( 1, &_texture );
    glBindTexture( GL_TEXTURE_2D, _texture );
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, 256, 1, 0, GL_RGB, GL_FLOAT, colors);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);  //Always set the base and max mipmap levels of a texture.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexEnvf(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE, GL_REPLACE);

	glEnable(GL_DEPTH_TEST);

	glClearColor(backColor[0], backColor[1], backColor[2], 1.0);

	glutMainLoop();
}
